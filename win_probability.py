import math
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# GeM Purchase Preference Constants
# --------------------------------------------------------

MSE_PREFERENCE = 0.15
MII_PREFERENCE = 0.20


def calculate_price_ratio(
    vendor_bid: float,
    predicted_l1: float,
) -> float:
    """
    Calculates how the vendor's quotation compares
    to the predicted winning price.

    ratio < 1.0
        Vendor is cheaper than expected L1.

    ratio = 1.0
        Vendor matches expected L1.

    ratio > 1.0
        Vendor is more expensive.
    """

    if vendor_bid <= 0:
        raise ValueError("Vendor bid must be positive.")

    if predicted_l1 <= 0:
        raise ValueError("Predicted L1 must be positive.")

    return vendor_bid / predicted_l1


def calculate_competition_factor(
    num_competitors: int,
) -> float:
    """
    Estimates the effect of competitor density.

    More competitors
        ↓

    Lower probability.

    We use a logarithmic decay because
    increasing from 3→6 competitors hurts
    much more than increasing from 30→33.
    """

    if num_competitors < 1:
        raise ValueError(
            "Competitor count must be at least 1."
        )

    return 1 / math.log2(num_competitors + 2)


def calculate_preference_bonus(
    is_mse: bool,
    is_mii: bool,
) -> float:
    """
    Returns the regulatory purchase
    preference bonus.

    We use the highest applicable
    preference.

    MII
        20%

    MSE
        15%
    """

    if is_mii:
        return MII_PREFERENCE

    if is_mse:
        return MSE_PREFERENCE

    return 0.0



def calculate_probability(
    price_ratio: float,
    competition_factor: float,
    preference_bonus: float,
) -> float:
    """
    Combines all scoring factors into a final probability.

    We use a logistic (sigmoid) function because:

    - Output is always between 0 and 100
    - Probability changes smoothly
    - Higher prices naturally reduce probability
    - Easy to interpret
    """

    # -------------------------------
    # Price score
    # -------------------------------
    #
    # ratio = 1.0  -> score = 0
    # ratio < 1.0  -> positive
    # ratio > 1.0  -> negative
    #
    # Weight = 8 because price is
    # the strongest factor.
    #

    price_score = 8 * (1 - price_ratio)

    # -------------------------------
    # Competition score
    # -------------------------------

    competition_score = 2 * competition_factor

    # -------------------------------
    # Preference score
    # -------------------------------

    preference_score = 2 * preference_bonus

    # -------------------------------
    # Final score
    # -------------------------------

    score = (
        price_score
        + competition_score
        + preference_score
    )

    probability = 100 / (1 + math.exp(-score))

    return round(probability, 2)

def calculate_win_probability(
    vendor_bid: float,
    predicted_l1: float,
    num_competitors: int,
    is_mse: bool = False,
    is_mii: bool = False,
) -> dict:
    """
    Complete Dynamic Win Probability Engine.
    """

    price_ratio = calculate_price_ratio(
        vendor_bid,
        predicted_l1,
    )

    competition_factor = calculate_competition_factor(
        num_competitors,
    )

    preference_bonus = calculate_preference_bonus(
        is_mse,
        is_mii,
    )

    probability = calculate_probability(
        price_ratio,
        competition_factor,
        preference_bonus,
    )

    if probability >= 80:
        verdict = "Very Strong"

    elif probability >= 60:
        verdict = "Strong"

    elif probability >= 40:
        verdict = "Moderate"

    elif probability >= 20:
        verdict = "Weak"

    else:
        verdict = "Very Weak"

    result = {

        "vendor_bid": float(vendor_bid),

        "predicted_l1": float(predicted_l1),

        "price_ratio": round(float(price_ratio), 3),

        "competition_factor": round(
            competition_factor,
            3,
        ),

        "preference_bonus": preference_bonus,

        "win_probability_pct": probability,

        "verdict": verdict,

    }

    logger.info(result)

    return result


def test_monotonicity():
    """
    Assignment Requirement:

    As vendor bid increases,
    win probability must strictly decrease.
    """

    predicted_l1 = 100000

    prices = [
        90000,
        95000,
        100000,
        105000,
        110000,
        115000,
        120000,
        125000,
        130000,
    ]

    probabilities = []

    print("\nMONOTONICITY TEST")
    print("-" * 45)

    for price in prices:

        result = calculate_win_probability(
            vendor_bid=price,
            predicted_l1=predicted_l1,
            num_competitors=8,
            is_mse=False,
            is_mii=False,
        )

        probabilities.append(result["win_probability_pct"])

        print(
            f"Bid ₹{price:>7,.0f}"
            f" -> "
            f"{result['win_probability_pct']:>6.2f}%"
        )

    # Verify monotonic decrease
    for i in range(len(probabilities) - 1):

        assert (
            probabilities[i] > probabilities[i + 1]
        ), (
            "Monotonicity failed: "
            f"{probabilities[i]} <= {probabilities[i+1]}"
        )

    print("\nPASS ✓ Probability decreases smoothly with increasing bid.")

def test_mse_bonus():
    """
    Vendors with MSE preference should receive
    a higher win probability than identical
    non-MSE vendors.
    """

    mse = calculate_win_probability(
        vendor_bid=105000,
        predicted_l1=100000,
        num_competitors=8,
        is_mse=True,
        is_mii=False,
    )

    normal = calculate_win_probability(
        vendor_bid=105000,
        predicted_l1=100000,
        num_competitors=8,
        is_mse=False,
        is_mii=False,
    )

    print("\nMSE BONUS TEST")
    print("-" * 45)

    print(
        f"MSE     : {mse['win_probability_pct']}%"
    )

    print(
        f"Non-MSE : {normal['win_probability_pct']}%"
    )

    assert (
        mse["win_probability_pct"]
        >
        normal["win_probability_pct"]
    )

    print("PASS ✓ MSE preference increases probability.")


def test_competitor_density():
    """
    More competitors should reduce
    win probability.
    """

    low = calculate_win_probability(
        vendor_bid=100000,
        predicted_l1=100000,
        num_competitors=3,
    )

    high = calculate_win_probability(
        vendor_bid=100000,
        predicted_l1=100000,
        num_competitors=20,
    )

    print("\nCOMPETITOR DENSITY TEST")
    print("-" * 45)

    print(
        f"3 competitors  : {low['win_probability_pct']}%"
    )

    print(
        f"20 competitors : {high['win_probability_pct']}%"
    )

    assert (
        low["win_probability_pct"]
        >
        high["win_probability_pct"]
    )

    print("PASS ✓ Competition reduces probability.")


if __name__ == "__main__":

    test_monotonicity()

    test_mse_bonus()

    test_competitor_density()