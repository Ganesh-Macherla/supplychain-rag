import re
from typing import Dict, List, Optional


SUPPLIER_SCORECARD = {
    "Nexa Polymers Ltd": {
        "on_time": 96.4,
        "defect_ppm": 320,
        "lead_time": 18,
        "spend": 12.4,
    },
    "Kaveri Metals Pvt Ltd": {
        "on_time": 88.1,
        "defect_ppm": 1150,
        "lead_time": 22,
        "spend": 8.7,
    },
    "Shenzhen Rui Electronics": {
        "on_time": 79.5,
        "defect_ppm": 210,
        "lead_time": 46,
        "spend": 21.9,
    },
    "Baltic Wire GmbH": {
        "on_time": 93.7,
        "defect_ppm": 90,
        "lead_time": 38,
        "spend": 15.2,
    },
    "Sunrise Connectors": {
        "on_time": 98.2,
        "defect_ppm": 140,
        "lead_time": 12,
        "spend": 6.1,
    },
    "Trident Circuit Boards": {
        "on_time": 84.6,
        "defect_ppm": 640,
        "lead_time": 34,
        "spend": 17.3,
    },
}


def verify_scorecard(question: str) -> Optional[Dict]:
    """
    Deterministically verifies factual questions that can be answered
    directly from the supplier scorecard.
    """
    q = question.lower()

    # Highest Q1 spend
    if (
        "highest" in q
        and "spend" in q
        and ("q1" in q or "quarter" in q)
    ):
        supplier, data = max(
            SUPPLIER_SCORECARD.items(),
            key=lambda item: item[1]["spend"],
        )

        return {
            "type": "highest_spend",
            "verified": True,
            "facts": {
                "supplier": supplier,
                "q1_spend_crore": data["spend"],
                "on_time_delivery": data["on_time"],
            },
            "statement": (
                f"{supplier} had the highest Q1 spend at "
                f"₹{data['spend']} crore, with "
                f"{data['on_time']}% on-time delivery."
            ),
        }

    # Supplier-specific defect rate
    for supplier, data in SUPPLIER_SCORECARD.items():
        if supplier.lower().replace(" pvt ltd", "") in q:
            if "defect" in q or "ppm" in q:
                return {
                    "type": "supplier_defect_rate",
                    "verified": True,
                    "facts": {
                        "supplier": supplier,
                        "defect_ppm": data["defect_ppm"],
                    },
                    "statement": (
                        f"{supplier} recorded "
                        f"{data['defect_ppm']} PPM."
                    ),
                }

    return None


def verify_policy_facts(
    supplier: str,
    on_time: Optional[float],
    defect_ppm: Optional[float],
    context: str,
) -> List[Dict]:
    """
    Deterministic verification of policy thresholds.

    The function only marks a condition as triggered when the required
    evidence is explicitly available.
    """
    results = []

    # Clause 6.1
    if on_time is not None:
        triggered = on_time < 90

        results.append(
            {
                "clause": "6.1",
                "status": "TRIGGERED" if triggered else "NOT TRIGGERED",
                "reason": (
                    f"{on_time}% is below the 90% threshold."
                    if triggered
                    else f"{on_time}% is not below the 90% threshold."
                ),
            }
        )

    # Clause 6.2 requires TWO consecutive quarters.
    # Do not infer the previous quarter merely because the current quarter
    # is below 85%.
    consecutive_evidence = bool(
        re.search(
            r"(second consecutive quarter|two consecutive quarters)",
            context,
            re.IGNORECASE,
        )
        and supplier.lower() in context.lower()
    )

    if on_time is not None:
        triggered = on_time < 85 and consecutive_evidence

        results.append(
            {
                "clause": "6.2",
                "status": "TRIGGERED" if triggered else "NOT CONFIRMED",
                "reason": (
                    "The documents establish two consecutive below-85% "
                    "quarters for this supplier."
                    if triggered
                    else (
                        "The documents do not establish two consecutive "
                        "below-85% quarters for this supplier."
                    )
                ),
            }
        )

    # Clause 6.3
    if defect_ppm is not None:
        triggered = defect_ppm > 500

        results.append(
            {
                "clause": "6.3",
                "status": "TRIGGERED" if triggered else "NOT TRIGGERED",
                "reason": (
                    f"{defect_ppm:,} PPM is above the 500 PPM threshold."
                    if triggered
                    else f"{defect_ppm:,} PPM is not above the 500 PPM threshold."
                ),
            }
        )

    return results