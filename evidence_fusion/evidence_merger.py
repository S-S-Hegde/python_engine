from typing import List, Dict, Any, Set

class EvidenceMerger:
    @staticmethod
    def deduplicate_and_group_evidence(raw_evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicates evidence objects by quote or location, preserving source traceability.
        """
        seen_quotes: Set[str] = set()
        merged: List[Dict[str, Any]] = []

        for ev in raw_evidence_list:
            quote = str(ev.get("quote") or ev.get("context") or ev.get("evidence_id") or "").strip()
            if not quote:
                continue

            quote_key = quote.lower()
            if quote_key in seen_quotes:
                # Merge source traceability into existing evidence item
                for existing in merged:
                    existing_quote = str(existing.get("quote") or existing.get("context") or "").strip().lower()
                    if existing_quote == quote_key:
                        existing_sources = existing.get("generated_from", [])
                        new_sources = ev.get("generated_from", [ev.get("location", "source")])
                        existing["generated_from"] = list(set(existing_sources + new_sources))
                        # Keep highest confidence score
                        existing["confidence"] = max(existing.get("confidence", 0.0), ev.get("confidence", 0.0))
                        break
            else:
                seen_quotes.add(quote_key)
                ev_copy = dict(ev)
                if "generated_from" not in ev_copy:
                    ev_copy["generated_from"] = [ev_copy.get("location", ev_copy.get("source", "Evidence"))]
                merged.append(ev_copy)

        return merged
