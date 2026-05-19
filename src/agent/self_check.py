def should_retry(results, answer):
    """
    Decide whether retrieval should retry.
    """

    distances = results["distances"][0]

    avg_distance = sum(distances) / len(distances)

    print(f"\nAverage Retrieval Distance: {avg_distance:.4f}")

    # ========================================================
    # Retry if retrieval quality weak
    # ========================================================

    if avg_distance > 0.55:

        print("Retry triggered: retrieval distances too high.")

        return True

    # ========================================================
    # Retry if model says insufficient evidence
    # ========================================================

    if "insufficient evidence" in answer.lower():

        print("Retry triggered: insufficient evidence detected.")

        return True

    return False


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    fake_results = {
        "distances": [[0.62, 0.59, 0.61]]
    }

    fake_answer = "Insufficient evidence in the provided documents."

    retry = should_retry(fake_results, fake_answer)

    print("\nShould Retry:", retry)