from app import load_and_split_document, create_vectorstore, create_qa_chain, ask_question
from test_questions import TEST_QUESTIONS
import datetime

def run_evaluation(pdf_path="data/WEF_Global_Cybersecurity_Outlook_2025.pdf"):
    print("Setting up pipeline...")
    chunks = load_and_split_document(pdf_path)
    vectorstore = create_vectorstore(chunks)
    retriever, llm = create_qa_chain(vectorstore)

    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\nRunning question {i}/{len(TEST_QUESTIONS)}: {question}")
        answer, sources = ask_question(retriever, llm, question)

        results.append({
            "question": question,
            "answer": answer,
            "num_sources_found": len(sources)
        })

    # Results ko file mein save karein
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Results\n\nRun date: {timestamp}\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"## Question {i}: {r['question']}\n\n")
            f.write(f"**Answer:** {r['answer']}\n\n")
            f.write(f"**Sources retrieved:** {r['num_sources_found']}\n\n")
            f.write("---\n\n")

    print("\nEvaluation complete! Results saved to evaluation_results.md")


if __name__ == "__main__":
    run_evaluation()