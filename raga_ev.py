import pandas as pd
from datasets import Dataset
import requests
import json
import re
import difflib
from collections import Counter


class MistralRAGASEvaluator:
    def __init__(self, ollama_url="http://localhost:11434"):
        """
        Initialize the evaluator with Ollama API URL

        Parameters:
        - ollama_url (str): URL for the Ollama API
        """
        self.ollama_url = ollama_url
        self.model = "mistral"  # Use the Mistral model

    def evaluate(self, ground_truth, answer, context=None):
        """
        Evaluates an answer against a ground truth using RAGAS-inspired metrics via Mistral LLM

        Parameters:
        - ground_truth (str): The correct answer
        - answer (str): The answer to evaluate
        - context (str or list, optional): The context/sources used for generating the answer

        Returns:
        - dict: Evaluation scores for various metrics
        """
        # Initialize scores dictionary
        scores = {}

        # If context is None, use ground truth as context
        if context is None:
            context = ground_truth
        elif isinstance(context, list):
            context = " ".join(context)

        # Add basic similarity score as fallback
        scores["string_similarity"] = self._string_similarity(ground_truth, answer)

        # Compute RAGAS-inspired metrics using Mistral
        ragas_scores = self._get_ragas_metrics(ground_truth, answer, context)
        scores.update(ragas_scores)

        # Calculate overall score (weighted average)
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "context_relevancy": 0.2,
            "context_recall": 0.2,
            "string_similarity": 0.0  # Only used as fallback
        }

        # Only consider metrics that we actually have
        valid_weights = {k: v for k, v in weights.items() if k in scores}
        total_weight = sum(valid_weights.values())

        if total_weight > 0:
            scores["overall_score"] = sum(scores[k] * valid_weights[k] for k in valid_weights) / total_weight
        else:
            scores["overall_score"] = scores["string_similarity"]

        return scores

    def _string_similarity(self, text1, text2):
        """Calculate string similarity using SequenceMatcher"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _get_ragas_metrics(self, ground_truth, answer, context):
        """
        Calculate RAGAS-inspired metrics using Mistral via Ollama
        """
        metrics = {}

        try:
            # 1. Faithfulness: Check if the answer is factually consistent with ground truth
            faithfulness_prompt = f"""Evaluate the factual consistency (faithfulness) of an answer compared to the ground truth.

Ground Truth: "{ground_truth}"

Answer: "{answer}"

Task: Score the faithfulness of the answer on a scale from 0.0 to 1.0, where:
- 1.0: The answer is completely factually consistent with the ground truth
- 0.5: The answer has some factual inconsistencies with the ground truth
- 0.0: The answer is completely inconsistent with the ground truth

Return ONLY a single number between 0.0 and 1.0 representing the faithfulness score.
"""

            # 2. Answer Relevancy: Check if the answer is relevant to what's being asked
            relevancy_prompt = f"""Evaluate the relevance of an answer to the ground truth.

Ground Truth: "{ground_truth}"

Answer: "{answer}"

Task: Score the relevance of the answer on a scale from 0.0 to 1.0, where:
- 1.0: The answer is perfectly relevant to what the ground truth is addressing
- 0.5: The answer is somewhat relevant but misses key points
- 0.0: The answer is completely irrelevant

Return ONLY a single number between 0.0 and 1.0 representing the relevance score.
"""

            # 3. Context Relevancy: Check if the context is relevant (if provided)
            context_relevancy_prompt = f"""Evaluate the relevance of the context to the ground truth.

Ground Truth: "{ground_truth}"

Context: "{context}"

Task: Score the relevance of the context on a scale from 0.0 to 1.0, where:
- 1.0: The context is perfectly relevant to the ground truth
- 0.5: The context is somewhat relevant but contains irrelevant information
- 0.0: The context is completely irrelevant

Return ONLY a single number between 0.0 and 1.0 representing the context relevance score.
"""

            # 4. Context Recall: Check how well the answer captures information from the context
            context_recall_prompt = f"""Evaluate how well the answer captures information from the context.

Context: "{context}"

Answer: "{answer}"

Task: Score the recall on a scale from 0.0 to 1.0, where:
- 1.0: The answer captures all relevant information from the context
- 0.5: The answer captures some but not all relevant information
- 0.0: The answer fails to capture relevant information from the context

Return ONLY a single number between 0.0 and 1.0 representing the recall score.
"""

            # Get faithfulness score
            faithfulness_score = self._query_mistral(faithfulness_prompt)
            if faithfulness_score is not None:
                metrics["faithfulness"] = faithfulness_score

            # Get answer relevancy score
            relevancy_score = self._query_mistral(relevancy_prompt)
            if relevancy_score is not None:
                metrics["answer_relevancy"] = relevancy_score

            # Get context relevancy score
            context_relevancy_score = self._query_mistral(context_relevancy_prompt)
            if context_relevancy_score is not None:
                metrics["context_relevancy"] = context_relevancy_score

            # Get context recall score
            context_recall_score = self._query_mistral(context_recall_prompt)
            if context_recall_score is not None:
                metrics["context_recall"] = context_recall_score

        except Exception as e:
            print(f"Warning: Error during RAGAS metric calculation: {str(e)}")
            metrics["error"] = str(e)

        return metrics

    def _query_mistral(self, prompt):
        """
        Query Mistral via Ollama API and extract a numerical score
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                response_text = response.json().get('response', '').strip()

                # Try to extract a number from the response
                number_match = re.search(r'(\d+\.\d+|\d+)', response_text)
                if number_match:
                    score = float(number_match.group(1))
                    return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
                else:
                    print(f"Warning: Could not extract score from response: {response_text}")
                    return None
            else:
                print(f"Warning: API request failed with status {response.status_code}")
                return None

        except Exception as e:
            print(f"Error querying Mistral: {str(e)}")
            return None

    def explain_scores(self, scores):
        """
        Provides an explanation of what each score means.

        Parameters:
        - scores (dict): The evaluation scores

        Returns:
        - str: Explanation of the scores
        """
        explanations = []

        # RAGAS metrics explanations
        if "faithfulness" in scores:
            explanations.append(f"Faithfulness Score: {scores['faithfulness']:.2f}/1.00")
            explanations.append("  Measures how factually consistent the answer is with the ground truth.")

        if "answer_relevancy" in scores:
            explanations.append(f"Answer Relevancy: {scores['answer_relevancy']:.2f}/1.00")
            explanations.append("  Measures how relevant the answer is to what the ground truth is addressing.")

        if "context_relevancy" in scores:
            explanations.append(f"Context Relevancy: {scores['context_relevancy']:.2f}/1.00")
            explanations.append("  Measures how relevant the provided context is to the ground truth.")

        if "context_recall" in scores:
            explanations.append(f"Context Recall: {scores['context_recall']:.2f}/1.00")
            explanations.append("  Measures how well the answer captures information from the context.")

        # Fallback metrics
        if "string_similarity" in scores and len(explanations) == 0:
            explanations.append(f"String Similarity: {scores['string_similarity']:.2f}/1.00")
            explanations.append("  Basic text similarity between ground truth and answer.")

        # Overall score
        if "overall_score" in scores:
            explanations.append(f"\nOverall Score: {scores['overall_score']:.2f}/1.00")

            # Qualitative assessment
            overall = scores['overall_score']
            if overall >= 0.9:
                explanations.append("Evaluation: Excellent match")
            elif overall >= 0.75:
                explanations.append("Evaluation: Good match")
            elif overall >= 0.6:
                explanations.append("Evaluation: Fair match")
            elif overall >= 0.4:
                explanations.append("Evaluation: Partial match")
            else:
                explanations.append("Evaluation: Poor match")

        # Error reporting
        if "error" in scores:
            explanations.append(f"\nNote: Encountered an error during evaluation: {scores['error']}")

        return "\n".join(explanations)


# Example usage
if __name__ == "__main__":
    evaluator = MistralRAGASEvaluator()

    print("=" * 70)
    print("RAGAS Evaluation with Mistral LLM")
    print("=" * 70)
    print("This evaluator uses Mistral through Ollama to calculate RAGAS metrics.")
    print("Make sure Ollama is running with the Mistral model loaded.")
    print("=" * 70)

    try:
        # Test if Ollama is available
        response = requests.get(evaluator.ollama_url + "/api/tags")
        if response.status_code != 200:
            raise Exception(f"Ollama API returned status code {response.status_code}")

        print("✓ Successfully connected to Ollama API.")

        # Check if model is available
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]

        if evaluator.model not in model_names and "mistral" not in model_names:
            print(f"Warning: '{evaluator.model}' model not found in Ollama. Available models: {model_names}")
            print("You may need to pull the model first with 'ollama pull mistral'")
            print("Continuing with basic similarity metrics only...")
    except Exception as e:
        print(f"Warning: Could not connect to Ollama API ({str(e)})")
        print("Make sure Ollama is running (https://ollama.ai/)")
        print("Continuing with basic similarity metrics only...")

    # Sample ground truth and answer
    print("\nExample evaluation:")
    ground_truth = "The capital of France is Paris."
    sample_answer = "Paris is the capital city of France."

    # Evaluate the answer
    scores = evaluator.evaluate(ground_truth, sample_answer)

    # Print the scores with explanations
    print("\nRAGAS Evaluation Results:\n")
    print(evaluator.explain_scores(scores))

    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Evaluation Mode")
    print("=" * 50)

    while True:
        print("\nEnter the ground truth (or 'quit' to exit):")
        gt = input("> ")
        if gt.lower() == 'quit':
            break

        print("Enter the answer to evaluate:")
        ans = input("> ")

        print("Enter context (optional, press Enter to skip):")
        ctx = input("> ")
        context = ctx if ctx else None

        scores = evaluator.evaluate(gt, ans, context)
        print("\nEvaluation Results:")
        print(evaluator.explain_scores(scores))