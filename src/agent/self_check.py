def should_retry(results,answer):
    """
    Decide whether retrieval should retry.
    """

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if not isinstance(results, dict):

        print("ERROR: results is not a dictionary.")
        print("Received type:", type(results))

        return False

    distances = results["distances"][0]
    
    avg_distance = sum(distances) / len(distances)

    print(f"\nAverage Retrieval Distance: {avg_distance:.4f}")

    # ========================================================
    # RETRY ONLY IF RETRIEVAL IS VERY BAD
    # ========================================================

    if avg_distance > 0.65:

        print("Retry triggered: retrieval distances too high.")

        return True

    # ========================================================
    # RETRY ONLY FOR STRONG FAILURE SIGNALS
    # ========================================================

    failure_phrases = [

        "no relevant information",
        "unable to answer",
        "documents do not contain",
        "context does not contain"
    ]

    answer_lower = answer.lower()

    for phrase in failure_phrases:

        if phrase in answer_lower:

            print(f"Retry triggered: detected '{phrase}'")

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