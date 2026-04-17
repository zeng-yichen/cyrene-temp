import os
import csv
import json
import re
import time
import requests
from backend.src.db import vortex as P
from datetime import datetime, timedelta, timezone

_MAX_ORDINAL_COMMENT_LEN = 10000
_WHY_POST_COMMENT_PREFIX = "Why we're posting this (internal):\n"

_LINKEDIN_MENTION_RE = re.compile(r"@\[([^\]]+)\]\(urn:li:\w+:\d+\)")


def _strip_linkedin_mentions(text: str) -> str:
    """Replace @[Name](urn:li:...) mention markup with plain-text Name."""
    return _LINKEDIN_MENTION_RE.sub(r"\1", text)


class Hyacinthia:
    """
    Client for pushing Amphoreus-generated drafts to the Ordinal platform.
    Parses the Stelle output structure, and natively prioritizes Cyrene's 
    rewritten markdown files if they exist.
    """
    def __init__(self, api_key=None):
        self.fallback_api_key = api_key or os.environ.get("ORDINAL_API_KEY")
        self.base_url = "https://app.tryordinal.com/api/v1"
        self.auth_csv_path = str(P.ordinal_auth_csv())

    def _update_story_inventory(
        self, company: str, post_id: str, post_text: str, publish_date: str
    ) -> None:
        """Append a USED record to story_inventory.md after confirmed Ordinal push.

        Only Hyacinthia writes USED records — Stelle never marks stories as used.
        This prevents rejected posts from permanently blocking stories.
        """
        inv_path = P.story_inventory_path(company)
        try:
            inv_path.parent.mkdir(parents=True, exist_ok=True)
            excerpt = post_text[:100].replace("\n", " ")
            line = f'\n- [USED — published {publish_date}] Ordinal ID: {post_id} | "{excerpt}..."\n'
            with open(inv_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            print(f"[Hyacinthia] Failed to update story inventory: {e}")

    def _save_draft_map_entry(
        self, company_keyword: str, post_id: str, content: str, title: str,
        generation_metadata: dict | None = None,
    ) -> bool:
        """Persist a post_id → original_text mapping to draft_map.json.

        This is THE critical registration point in the draft preservation pipeline:
        every post Stelle pushes to Ordinal must land here, keyed by the Ordinal
        workspace post id, so that when ordinal_sync later ingests the published
        version it can compute the (draft, published) delta.

        Bulletproof contract: never raises. Returns True on successful persistence,
        False on any failure. Logs every success AND every failure with enough
        context to grep the sync logs for pairing issues.

        Input validation: an empty post_id or content is treated as a bug (logged
        loudly) — these are both essential for the pipeline to work and silent
        acceptance was the root cause of past 'it looked like it was working'
        failures.
        """
        # --- input validation: loud failures on bad inputs ---
        pid = (post_id or "").strip()
        text = (content or "").strip()
        if not pid:
            print(
                f"[ORDINAL CLIENT] DRAFT_MAP REGISTRATION FAILED ({company_keyword}): "
                f"empty post_id — the draft→published pairing pipeline will be broken "
                f"for this post. Title: {title!r}"
            )
            return False
        if not text:
            print(
                f"[ORDINAL CLIENT] DRAFT_MAP REGISTRATION FAILED ({company_keyword}/{pid[:12]}): "
                f"empty content — the draft→published pairing pipeline will be broken "
                f"for this post. Title: {title!r}"
            )
            return False

        path = P.draft_map_path(company_keyword)

        # --- load existing map (permissive on read errors) ---
        draft_map: dict = {}
        if path.exists():
            try:
                draft_map = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(draft_map, dict):
                    print(
                        f"[ORDINAL CLIENT] draft_map.json for {company_keyword} "
                        f"was not a dict ({type(draft_map).__name__}) — resetting to empty"
                    )
                    draft_map = {}
            except Exception as e:
                print(
                    f"[ORDINAL CLIENT] Could not read draft_map.json for {company_keyword}: "
                    f"{e} — starting fresh"
                )
                draft_map = {}

        entry = {
            "original_text": text,
            "title": title,
            "company": company_keyword,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Attach generation-time quality metadata for ordinal_sync to pick up
        if generation_metadata:
            for k in ("cyrene_composite", "cyrene_dimensions", "cyrene_dimension_set",
                       "cyrene_iterations", "cyrene_weights_tier",
                       "constitutional_results", "alignment_score"):
                if k in generation_metadata:
                    entry[k] = generation_metadata[k]
        draft_map[pid] = entry

        # --- persist (loud failure on write errors) ---
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(draft_map, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            print(
                f"[ORDINAL CLIENT] DRAFT_MAP REGISTRATION FAILED ({company_keyword}/{pid[:12]}): "
                f"write failed: {e} — the draft→published pairing pipeline will be "
                f"broken for this post"
            )
            return False

        print(
            f"[ORDINAL CLIENT] draft_map registered: {company_keyword}/{pid[:12]}… "
            f"({len(text)} chars, total entries: {len(draft_map)})"
        )
        return True

    def remove_draft_map_entry(self, company_keyword: str, ordinal_post_id: str) -> None:
        """Drop a stale Ordinal post id from draft_map.json (e.g. local draft was re-pushed)."""
        oid = (ordinal_post_id or "").strip()
        if not oid:
            return
        path = P.draft_map_path(company_keyword)
        try:
            draft_map = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            draft_map = {}
        if oid not in draft_map:
            return
        del draft_map[oid]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(draft_map, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"[ORDINAL CLIENT] Failed to update draft_map.json: {e}")

    def upload_asset_from_public_url(
        self,
        company_keyword: str,
        file_url: str,
        poll_interval: float = 2.0,
        max_wait_seconds: float = 120.0,
    ) -> str | None:
        """Ordinal downloads from URL; poll until assetId is ready. Returns asset UUID or None."""
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            print(f"[ORDINAL CLIENT] No API key for upload_asset_from_public_url ({company_keyword})")
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(
                f"{self.base_url}/uploads",
                headers=headers,
                json={"url": file_url},
                timeout=60,
            )
            r.raise_for_status()
            job = r.json()
            job_id = job.get("id")
            if not job_id:
                print(f"[ORDINAL CLIENT] Upload response missing id: {job}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[ORDINAL CLIENT] Upload POST failed: {e}")
            return None

        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            try:
                sr = requests.get(
                    f"{self.base_url}/uploads/{job_id}",
                    headers=headers,
                    timeout=30,
                )
                sr.raise_for_status()
                st = sr.json()
                status = (st.get("status") or "").lower()
                if status == "ready":
                    aid = st.get("assetId")
                    if aid:
                        return str(aid)
                    print(f"[ORDINAL CLIENT] Upload ready but no assetId: {st}")
                    return None
                if status in ("failed", "expired"):
                    print(f"[ORDINAL CLIENT] Upload {status}: {st.get('error', st)}")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"[ORDINAL CLIENT] Upload poll failed: {e}")
                return None
            time.sleep(poll_interval)

        print(f"[ORDINAL CLIENT] Upload timed out waiting for asset (job {job_id})")
        return None

    def _get_api_key_for_client(self, company_keyword: str) -> str:
        """Reads the CSV file to find the specific API key for the company."""
        if not os.path.exists(self.auth_csv_path):
            return self.fallback_api_key
            
        try:
            with open(self.auth_csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    c_id = row.get("company_id", "").strip().lower()
                    slug = row.get("provider_org_slug", "").strip().lower()
                    target = company_keyword.strip().lower()
                    
                    if c_id == target or slug == target:
                        key = row.get("api_key", "").strip()
                        if key:
                            return key
        except Exception as e:
            print(f"[ORDINAL CLIENT WARNING] Failed to read auth CSV: {e}")
            
        return self.fallback_api_key

    def _post_castorice_thread_comments(
        self, company_keyword: str, post_id: str, cr: dict | None
    ) -> None:
        """Post Castorice citation comments and optional why-post blurb as Ordinal thread comments."""
        if not post_id or not isinstance(cr, dict):
            return
        for cite_comment in cr.get("citation_comments") or []:
            if not cite_comment:
                continue
            c_res = self.create_comment(company_keyword, post_id, str(cite_comment))
            if not c_res.get("success"):
                print(f"[ORDINAL CLIENT WARNING] Citation comment failed: {c_res.get('error')}")
        why = (cr.get("why_post") or "").strip()
        if why:
            msg = _WHY_POST_COMMENT_PREFIX + why
            if len(msg) > _MAX_ORDINAL_COMMENT_LEN:
                msg = msg[: _MAX_ORDINAL_COMMENT_LEN - 3] + "..."
            w_res = self.create_comment(company_keyword, post_id, msg)
            if not w_res.get("success"):
                print(f"[ORDINAL CLIENT WARNING] Why-post comment failed: {w_res.get('error')}")

    def parse_rewritten_posts(self, content: str) -> list:
        """
        Parses Cyrene's generated _rewritten_posts.md file.
        Safely isolates the actual text of the finalized post, accounting for 
        various Markdown formatting quirks (headers, bolding, XML tags).
        """
        posts_to_upload = []
        
        # Split the markdown file by Post delineations (e.g., # POST 1, ## POST 2 REWRITE)
        blocks = re.split(r'(?i)(?:#+\s*POST\s*\d+|\[POST:?\s*\d+\])', content)
        
        # We start enumerating from 1 since the 0th block is usually the preamble
        post_index = 1 
        
        for block in blocks:
            if not block.strip() or "Fact Extraction" not in block:
                continue
                
            theme = f"Cyrene Edited Draft {post_index}"
            final_post_text = ""
            
            # 1. Attempt to locate the exact markdown section for the Final Post
            # This updated regex catches: 
            # "## Final Post", "**FINAL STYLIZED POST**", "### Final Rewritten Post:"
            match = re.search(
                r'(?i)(?:#+\s*|\*\*\s*)Final\s*(?:Rewritten\s*|Stylized\s*|Edited\s*)?Post(?:\s*\*\*|:)?\s*\n(.*?)(?:\*{10,}|-{10,}|\Z)', 
                block, 
                re.DOTALL
            )
            
            if match:
                final_post_text = match.group(1).strip()
            else:
                # 2. Fallback: If Cyrene's raw XML tags accidentally leaked into the written file
                xml_match = re.search(r'<final_post>(.*?)</final_post>', block, re.DOTALL | re.IGNORECASE)
                if xml_match:
                    final_post_text = xml_match.group(1).strip()
                else:
                    # 3. Final Fallback: Split by the last bold or header marker
                    # Looks for the last bolded section (e.g. **Step 3**) or Markdown header
                    sections = re.split(r'\n(?:#+\s+|\*\*)', block)
                    if len(sections) > 1:
                        # Extract everything after the last section title
                        final_post_text = sections[-1].split('\n', 1)[-1].strip()
                    else:
                        final_post_text = block.strip()
            
            # Clean up any trailing horizontal rules, separator asterisks, or stray bold marks
            final_post_text = re.sub(r'\n\*{5,}.*', '', final_post_text, flags=re.DOTALL).strip()
            final_post_text = re.sub(r'\n-{5,}.*', '', final_post_text, flags=re.DOTALL).strip()
            # Remove any trailing bold asterisks if the regex caught them
            final_post_text = re.sub(r'\*\*\s*$', '', final_post_text).strip()

            if final_post_text:
                posts_to_upload.append({
                    "theme": theme,
                    "content": final_post_text
                })
                post_index += 1
                
        return posts_to_upload

    def parse_posts(self, content: str) -> list:
        """
        Parses the Stelle base output text to extract individual posts and their 
        corrected versions (if Permansor Terrae applied any fixes).
        """
        posts_to_upload = []
        
        drafts_marker = "--- FINAL LINKEDIN POST DRAFTS ---"
        if drafts_marker in content:
            content = content.split(drafts_marker)[-1]
            
        post_blocks = re.split(r'POST \d+ THEME:', content)
        
        for block in post_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            theme = lines[0].strip()
            
            if "### 🛡️ Fact-Check Report" in block:
                parts = block.split("### 🛡️ Fact-Check Report")
                draft_section, report_section = parts[0], parts[1]
                
                original_post = ""
                if "-------------------------" in draft_section:
                    original_post = draft_section.split("-------------------------", 1)[1].strip()
                
                final_post_text = original_post
                
                if "[CORRECTED POST]:" in report_section:
                    final_post_text = report_section.split("[CORRECTED POST]:", 1)[1]
                elif "[CORRECTED POST]" in report_section:
                    final_post_text = report_section.split("[CORRECTED POST]", 1)[1]
                    
                final_post_text = final_post_text.split("*" * 50)[0].strip()
                
                if final_post_text:
                    posts_to_upload.append({"theme": theme, "content": final_post_text})
            else:
                draft_body = block
                if "-------------------------" in block:
                    draft_body = block.split("-------------------------", 1)[1]
                
                draft_body = draft_body.split("*" * 50)[0].strip()
                if draft_body:
                    posts_to_upload.append({"theme": theme, "content": draft_body})
                    
        if not posts_to_upload:
            posts_to_upload.append({"theme": "Extracted Drafts", "content": content.strip()})
            
        return posts_to_upload

    def _compute_publish_dates(self, start_date: datetime, num_posts: int, posts_per_month: int) -> list:
        """
        Compute publish dates based on client's posting frequency.
        
        Args:
            start_date: The first date to consider for scheduling
            num_posts: Number of posts to schedule
            posts_per_month: Either 12 (Mon/Wed/Thu) or 8 (Tue/Thu)
            
        Returns:
            List of datetime objects for each post's publish date
        """
        if posts_per_month == 12:
            # Monday=0, Wednesday=2, Thursday=3
            valid_weekdays = {0, 2, 3}
        else:
            # Tuesday=1, Thursday=3
            valid_weekdays = {1, 3}
        
        publish_dates = []
        current_date = start_date
        
        while len(publish_dates) < num_posts:
            if current_date.weekday() in valid_weekdays:
                publish_dates.append(current_date)
            current_date += timedelta(days=1)
        
        return publish_dates

    def push_drafts(
        self,
        company_keyword: str,
        model_name: str,
        content: str,
        posts_per_month: int = 12,
        start_date: datetime = None,
        per_post: list = None,
        castorice_results: list = None,
        schedule_publish_at: datetime | None = None,
        default_approvals: list | None = None,
        prefer_rewritten_file: bool = True,
        inline_single_post: bool = False,
        single_post_title: str | None = None,
    ) -> tuple:
        """
        Parses the file and iteratively pushes each drafted post to the Ordinal workspace.
        Automatically intercepts and prioritizes Cyrene's edited Markdown file if present.
        
        Args:
            company_keyword: Client identifier
            model_name: Name of the model used
            content: Raw content to parse
            posts_per_month: 12 for Mon/Wed/Thu scheduling, 8 for Tue/Thu scheduling
            start_date: First date to consider for scheduling (defaults to 7 days from now)
            per_post: Optional list aligned with parsed posts; each entry may be a dict with
                optional keys: label_ids (list of UUID str), approvals (list of dicts for
                create_approvals: userId, optional message, dueDate, isBlocking).
            schedule_publish_at: If set, each post uses this moment as publishAt (UTC-aware
                or naive treated as UTC in strftime) instead of computed cadence slots.
            default_approvals: Used when per_post[i] has no approvals entry; same list
                applied to every created post in this batch.
            prefer_rewritten_file: If True and Cyrene's _rewritten_posts.md exists, parse that
                file instead of `content` (batch desktop workflow). Set False for API single-draft push.
            inline_single_post: If True, push exactly one post from `content` (no file read, no parse_posts).
            single_post_title: Optional title/theme when inline_single_post is True.
        """
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            return (
                False,
                f"API key not found for '{company_keyword}' in CSV and fallback ORDINAL_API_KEY is not set.",
                [],
            )

        endpoint = f"{self.base_url}/posts"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        rewritten_filepath = str(P.post_dir(company_keyword) / f"{company_keyword}_rewritten_posts.md")

        if inline_single_post:
            body = (content or "").strip()
            if not body:
                return False, "No content to push.", []
            theme = (single_post_title or "").strip() or "Draft"
            parsed_posts = [{"theme": theme[:200], "content": body}]
            print(f"\n[ORDINAL CLIENT] Single inline draft push ({len(body)} chars) for {company_keyword}...")
        elif prefer_rewritten_file and os.path.exists(rewritten_filepath):
            print(f"\n[ORDINAL CLIENT] Overriding base text. Found Cyrene edits at: {rewritten_filepath}")
            with open(rewritten_filepath, "r", encoding="utf-8") as f:
                rewritten_content = f.read()
            parsed_posts = self.parse_rewritten_posts(rewritten_content)
            model_name = "Cyrene Edit"
        else:
            parsed_posts = self.parse_posts(content)
        
        print(f"\n[ORDINAL CLIENT] Parsed {len(parsed_posts)} {model_name} draft(s). Pushing to Ordinal API for {company_keyword}...")
        
        # Compute publish dates — use Temporal Orchestrator when available,
        # fall back to fixed cadence logic otherwise.
        if start_date is None:
            start_date = datetime.utcnow() + timedelta(days=7)

        _use_orchestrator = False
        publish_dates = []
        try:
            from backend.src.services.temporal_orchestrator import compute_publish_dates_optimized
            publish_dates = compute_publish_dates_optimized(
                company_keyword, len(parsed_posts), start_date,
            )
            if publish_dates:
                _use_orchestrator = True
                print(f"[ORDINAL CLIENT] Using Temporal Orchestrator scheduling for {company_keyword}")
        except Exception as _to_err:
            print(f"[ORDINAL CLIENT] Temporal Orchestrator unavailable, using fixed cadence: {_to_err}")

        if not publish_dates:
            publish_dates = self._compute_publish_dates(start_date, len(parsed_posts), posts_per_month)
        weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        schedule_type = "Mon/Wed/Thu" if posts_per_month == 12 else "Tue/Thu"
        if schedule_publish_at is not None:
            print(
                f"[ORDINAL CLIENT] Scheduling {len(parsed_posts)} post(s) at "
                f"{schedule_publish_at.strftime('%Y-%m-%dT%H:%M:%S')} (UTC) publishAt"
            )
        else:
            print(
                f"[ORDINAL CLIENT] Scheduling {len(parsed_posts)} posts on {schedule_type} "
                f"starting from {start_date.strftime('%Y-%m-%d')}"
            )
        
        success_count = 0
        error_msgs = []
        first_url = None
        created_ordinal_ids: list[str] = []

        # Iteratively post each draft
        shared_approvals = list(default_approvals) if default_approvals else []

        for i, post_data in enumerate(parsed_posts):
            if schedule_publish_at is not None:
                pub_date = schedule_publish_at
                tentative_date = schedule_publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            elif _use_orchestrator:
                pub_date = publish_dates[i]
                tentative_date = pub_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            else:
                pub_date = publish_dates[i]
                tentative_date = pub_date.strftime("%Y-%m-%dT09:00:00.000Z")

            extras = {}
            if per_post and i < len(per_post) and isinstance(per_post[i], dict):
                extras = per_post[i]
            label_ids = extras.get("label_ids") or []
            if "approvals" in extras:
                post_approvals = extras["approvals"] or []
            else:
                post_approvals = list(shared_approvals)
            
            # --- NEW: Create title from the first 5 words of the post content ---
            content_words = post_data["content"].split()
            if len(content_words) > 5:
                clean_title = " ".join(content_words[:5]) + "..."
            else:
                clean_title = " ".join(content_words)
                
            # Fallback just in case the content is empty
            if not clean_title.strip():
                clean_title = f"Draft {i+1}"
            
            # Strip LinkedIn mention markers BEFORE sending to Ordinal AND before
            # writing to draft_map. Storing the stripped version ensures the
            # draft_map text matches what Ordinal actually received, so later
            # fuzzy-matching against the published body produces high similarity.
            stripped_content = _strip_linkedin_mentions(post_data["content"])
            linked_in_cfg: dict = {"copy": stripped_content}
            li_assets = extras.get("linkedin_asset_ids") or []
            if li_assets:
                linked_in_cfg["assetIds"] = li_assets

            payload = {
                "title": clean_title,
                "publishAt": tentative_date,
                "status": "InProgress",
                "linkedIn": linked_in_cfg,
            }
            if label_ids:
                payload["labelIds"] = label_ids
            
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status() 
                
                data = response.json()
                post_id = data.get("id")
                if not first_url:
                    first_url = data.get("url")
                    
                success_count += 1
                if post_id:
                    created_ordinal_ids.append(str(post_id))
                if schedule_publish_at is not None:
                    print(
                        f"[ORDINAL CLIENT] Success! Post {i+1} created "
                        f"(publishAt={tentative_date}) ID: {post_id}"
                    )
                else:
                    day_name = weekday_names[pub_date.weekday()]
                    print(
                        f"[ORDINAL CLIENT] Success! Post {i+1} created "
                        f"(scheduled {day_name} {pub_date.strftime('%Y-%m-%d')}) ID: {post_id}"
                    )

                if post_id:
                    # Store the stripped content — what Ordinal actually received —
                    # so the draft_map text aligns with what ingest_from_ordinal
                    # will see in the analytics feed. Using the raw post_data
                    # here would leave LinkedIn mention markers in the stored
                    # text, dropping fuzzy-match similarity against the
                    # published body.
                    self._save_draft_map_entry(
                        company_keyword, post_id, stripped_content, clean_title
                    )
                    self._update_story_inventory(
                        company_keyword, post_id, stripped_content, tentative_date
                    )

                if post_approvals and post_id:
                    app_res = self.create_approvals(company_keyword, post_id, post_approvals)
                    if not app_res.get("success"):
                        print(f"[ORDINAL CLIENT WARNING] Post {i+1} approvals failed: {app_res.get('error')}")

                if castorice_results and post_id and i < len(castorice_results):
                    cr = castorice_results[i]
                    self._post_castorice_thread_comments(company_keyword, post_id, cr)
                
            except requests.exceptions.HTTPError as e:
                error_msg = str(e)
                try:
                    error_data = response.json()
                    error_msg += f"\nDetails: {json.dumps(error_data.get('data', error_data), indent=2)}"
                except Exception:
                    pass
                print(f"[ORDINAL CLIENT ERROR] Post {i+1} failed: {error_msg}")
                error_msgs.append(f"Post {i+1} failed: {error_msg}")
            except Exception as e:
                print(f"[ORDINAL CLIENT ERROR] Post {i+1} failed: {e}")
                error_msgs.append(f"Post {i+1} failed: {e}")

        if success_count > 0:
            return True, first_url, created_ordinal_ids
        return False, "\n".join(error_msgs), []

    def get_recent_comments(self, company_keyword: str, publish_date_min: str) -> dict: 
        """
        Fetches all standard AND inline comments from posts scheduled on or after a specific date.
        If a post has comments, it fetches the original post content and pairs them together, 
        saving a .txt file in the client's feedback directory.
        
        Returns:
            A dictionary containing lists of 'standard_comments' and 'inline_comments'.
        """
        api_key = self._get_api_key_for_client(company_keyword)
        
        if not api_key:
            print(f"[EVERNIGHT ERROR] API key not found for '{company_keyword}'.")
            return {"standard_comments": [], "inline_comments": []}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Ensure the feedback directory exists
        feedback_dir = str(P.feedback_dir(company_keyword))
        os.makedirs(feedback_dir, exist_ok=True)
        
        # STEP 1: Fetch all Post IDs scheduled after the target date
        post_ids = []
        has_more = True
        cursor = None
        
        print(f"\n[EVERNIGHT] Fetching posts for {company_keyword} scheduled on/after {publish_date_min}...")
        
        while has_more:
            params = {"limit": 100, "publishDateMin": publish_date_min}
            if cursor:
                params["cursor"] = cursor
                
            try:
                response = requests.get(f"{self.base_url}/posts", headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                for post in data.get("posts", []):
                    post_ids.append(post.get("id"))
                    
                has_more = data.get("hasMore", False)
                cursor = data.get("nextCursor")
                
            except Exception as e:
                print(f"[EVERNIGHT ERROR] Failed during post retrieval: {e}")
                has_more = False
                
        if not post_ids:
            return {"standard_comments": [], "inline_comments": []}
            
        # STEP 2: Fetch Comments and Original Post text
        all_standard_comments = []
        all_inline_comments = []
        
        for post_id in post_ids:
            post_standard_comments = []
            post_inline_comments = []
            
            # A. Fetch Standard Comments
            try:
                response = requests.get(f"{self.base_url}/posts/{post_id}/comments", headers=headers)
                if response.status_code == 200:
                    comments_data = response.json().get("comments", [])
                    for comment in comments_data:
                        comment["target_postId"] = post_id
                        post_standard_comments.append(comment)
                        all_standard_comments.append(comment)
            except Exception as e:
                print(f"[EVERNIGHT WARNING] Failed to fetch standard comments for post {post_id}: {e}")
                
            # B. Fetch Inline (Text-Anchored) Comments
            try:
                response = requests.get(f"{self.base_url}/posts/{post_id}/inline-comments", headers=headers)
                if response.status_code == 200:
                    inline_data = response.json().get("inlineComments", [])
                    for inline_comment in inline_data:
                        inline_comment["target_postId"] = post_id
                        post_inline_comments.append(inline_comment)
                        all_inline_comments.append(inline_comment)
            except Exception as e:
                print(f"[EVERNIGHT WARNING] Failed to fetch inline comments for post {post_id}: {e}")
                
            # C. If comments exist, fetch the post text and save the paired document
            if post_standard_comments or post_inline_comments:
                post_text = "[Original post content could not be retrieved]"
                post_title = f"Post_{post_id}"
                
                try:
                    p_res = requests.get(f"{self.base_url}/posts/{post_id}", headers=headers)
                    if p_res.status_code == 200:
                        post_data = p_res.json().get("post", {})
                        post_title = post_data.get("title", post_title)
                        
                        # Prioritize LinkedIn copy, fallback to X/Twitter
                        if post_data.get("linkedIn") and post_data["linkedIn"].get("copy"):
                            post_text = post_data["linkedIn"]["copy"]
                        elif post_data.get("x") and post_data["x"].get("tweets"):
                            post_text = "\n".join([t.get("copy", "") for t in post_data["x"]["tweets"]])
                except Exception as e:
                    print(f"[EVERNIGHT WARNING] Failed to fetch post content for {post_id}: {e}")

                # Save the formatted feedback file
                safe_title = "".join([c if c.isalnum() else "_" for c in post_title])[:50]
                filepath = os.path.join(feedback_dir, f"feedback_{safe_title}_{post_id[:8]}.txt")
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"=== ORIGINAL POST ===\n{post_text}\n\n")
                    
                    if post_standard_comments:
                        f.write("=== GENERAL COMMENTS ===\n")
                        for c in post_standard_comments:
                            user = c.get("user", {})
                            author = f"{user.get('firstName', '')} {user.get('lastName', '')}".strip()
                            f.write(f"[{author}]: {c.get('message')}\n")
                        f.write("\n")
                        
                    if post_inline_comments:
                        f.write("=== INLINE COMMENTS (Text-Anchored) ===\n")
                        for ic in post_inline_comments:
                            resolved_tag = "[RESOLVED] " if ic.get("resolved") else ""
                            highlighted = ic.get("highlightedText", "Unknown text")
                            f.write(f"{resolved_tag}Highlight: \"{highlighted}\"\n")
                            
                            for reply in ic.get("replies", []):
                                user = reply.get("user", {})
                                author = f"{user.get('firstName', '')} {user.get('lastName', '')}".strip()
                                f.write(f"  -> [{author}]: {reply.get('message')}\n")
                            f.write("\n")
                            
                print(f"[EVERNIGHT] Saved paired feedback document for Post: {post_title}")
                
        print(f"[EVERNIGHT] Retrieved {len(all_standard_comments)} standard, {len(all_inline_comments)} inline comment(s).")
        return {
            "standard_comments": all_standard_comments,
            "inline_comments": all_inline_comments
        }

    def get_labels(self, company_keyword: str) -> list:
        """
        Fetch all labels from the workspace.
        
        Returns:
            List of label dicts with id, name, color, backgroundColor
        """
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            print(f"[EVERNIGHT ERROR] API key not found for '{company_keyword}'.")
            return []
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.base_url}/labels", headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("labels", [])
        except Exception as e:
            print(f"[EVERNIGHT ERROR] Failed to fetch labels: {e}")
            return []

    def get_users(self, company_keyword: str) -> list:
        """
        Fetch all users from the workspace (potential approvers).
        
        Returns:
            List of user dicts with id, email, firstName, lastName
        """
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            print(f"[EVERNIGHT ERROR] API key not found for '{company_keyword}'.")
            return []
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{self.base_url}/users", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[EVERNIGHT ERROR] Failed to fetch users: {e}")
            return []

    def push_single_post(
        self,
        company_keyword: str,
        content: str,
        publish_date: datetime,
        status: str = "InProgress",
        label_ids: list = None,
        title: str = None,
        approvals: list = None,
        castorice_result: dict = None,
        linkedin_asset_ids: list | None = None,
        generation_metadata: dict | None = None,
    ) -> dict:
        """
        Push a single post to Ordinal.
        
        Args:
            company_keyword: Client identifier
            content: LinkedIn post copy
            publish_date: Datetime for scheduled publish
            status: Post status (Tentative, ToDo, InProgress, ForReview, Blocked, Finalized, Scheduled); default InProgress
            label_ids: Optional list of label UUIDs to attach
            title: Optional title (defaults to first 5 words of content)
            approvals: Optional list of approval dicts (userId, optional message, dueDate, isBlocking);
                submitted after the post is created.
            linkedin_asset_ids: Optional Ordinal asset UUIDs for LinkedIn image attachments.
            
        Returns:
            Dict with 'success', 'post_id', 'url', 'error' keys
        """
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            return {"success": False, "error": f"API key not found for '{company_keyword}'"}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        if not title:
            content_words = content.split()
            if len(content_words) > 5:
                title = " ".join(content_words[:5]) + "..."
            else:
                title = " ".join(content_words) if content_words else "Untitled Draft"
        
        publish_at = publish_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

        content = _strip_linkedin_mentions(content)
        linked_in_cfg: dict = {"copy": content}
        if linkedin_asset_ids:
            linked_in_cfg["assetIds"] = linkedin_asset_ids

        payload = {
            "title": title,
            "publishAt": publish_at,
            "status": status,
            "linkedIn": linked_in_cfg,
        }
        
        if label_ids:
            payload["labelIds"] = label_ids
        
        try:
            response = requests.post(f"{self.base_url}/posts", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            post_id = data.get("id")
            if approvals and post_id:
                app_res = self.create_approvals(company_keyword, post_id, approvals)
                if not app_res.get("success"):
                    print(f"[ORDINAL] Approval request failed: {app_res.get('error')}")
            if post_id:
                self._save_draft_map_entry(company_keyword, post_id, content,
                                           title or content.split()[0],
                                           generation_metadata=generation_metadata)
                self._update_story_inventory(company_keyword, post_id, content, publish_at)
            if post_id:
                self._post_castorice_thread_comments(company_keyword, post_id, castorice_result)
            return {
                "success": True,
                "post_id": post_id,
                "url": data.get("url"),
                "error": None
            }
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_data = response.json()
                error_msg += f" - {json.dumps(error_data.get('data', error_data))}"
            except Exception:
                pass
            return {"success": False, "post_id": None, "url": None, "error": error_msg}
        except Exception as e:
            return {"success": False, "post_id": None, "url": None, "error": str(e)}

    def create_approvals(
        self,
        company_keyword: str,
        post_id: str,
        approvals: list,
    ) -> dict:
        """
        Create approval requests for a post.
        
        Args:
            company_keyword: Client identifier
            post_id: The UUID of the post
            approvals: List of dicts with 'userId', optional 'message', 'dueDate', 'isBlocking'
            
        Returns:
            Dict with 'success', 'created', 'existing', 'error' keys
        """
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            return {"success": False, "error": f"API key not found for '{company_keyword}'"}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "postId": post_id,
            "approvals": approvals
        }
        
        try:
            response = requests.post(f"{self.base_url}/approvals", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "created": data.get("createdApprovals", []),
                "existing": data.get("existingApprovals", []),
                "error": None
            }
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_data = response.json()
                error_msg += f" - {json.dumps(error_data.get('data', error_data))}"
            except Exception:
                pass
            return {"success": False, "created": [], "existing": [], "error": error_msg}
        except Exception as e:
            return {"success": False, "created": [], "existing": [], "error": str(e)}

    def create_comment(
        self,
        company_keyword: str,
        post_id: str,
        message: str,
    ) -> dict:
        """
        Create a comment on a post.
        
        Args:
            company_keyword: Client identifier
            post_id: The UUID of the post
            message: The comment text
            
        Returns:
            Dict with 'success', 'comment_id', 'error' keys
        """
        api_key = self._get_api_key_for_client(company_keyword)
        if not api_key:
            return {"success": False, "error": f"API key not found for '{company_keyword}'"}
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"message": message}
        
        try:
            response = requests.post(
                f"{self.base_url}/posts/{post_id}/comments",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "comment_id": data.get("id"),
                "error": None
            }
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_data = response.json()
                error_msg += f" - {json.dumps(error_data.get('data', error_data))}"
            except Exception:
                pass
            return {"success": False, "comment_id": None, "error": error_msg}
        except Exception as e:
            return {"success": False, "comment_id": None, "error": str(e)}
