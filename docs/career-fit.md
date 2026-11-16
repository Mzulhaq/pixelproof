# Career fit: Python backend and applied AI foundations

PixelProof shows a practical slice of backend and applied ML engineering: validating a model artifact, keeping training dependencies out of the serving path, bounding an API request, and documenting a fixed evaluation. It can support a conversation about model packaging and inference mechanics. It does not establish role eligibility, professional experience, or production ML operating experience.

## Current role examples checked November 16, 2026

Capgemini lists a **Junior Python Developer** position in Iasi. The posting describes Python components, APIs and backend services, tests, documentation, and deployment support. It also asks for at least two years of overall IT experience, including one year of hands-on Python development, and French and English. That makes the description a useful skills reference for backend practice, but the title alone does not establish that a particular candidate meets its experience or language requirements. Review the current [Capgemini posting](https://careers.capgemini.com/job/Iasi-Junior-Python-Developer/1427338033/) before applying.

Showpad lists an **AI Engineer** position in Bucharest involving Python, LLMs, and cloud platforms; the posting names RAG, search algorithms, and evaluation among preferred experience. PixelProof is relevant to the foundations of packaging and serving a trained classifier. It does not integrate an LLM, embeddings, RAG, or a cloud inference service, and it does not establish the hands-on AI/LLM experience listed in the role requirements. See the current [Showpad posting](https://job-boards.greenhouse.io/showpad/jobs/8264241).

## How to present the project

Describe the model as a trained logistic regression classifier with a pure-Python serving implementation. Explain why the held-out data is a resplit of scikit-learn's bundled copy of the UCI test set, and why it is not an official benchmark or writer-independent evaluation. Discuss artifact validation and the narrow purpose of its checksum, then explain uncalibrated probabilities and the lack of out-of-distribution detection. Report only measured metrics from the generated evaluation artifact; do not imply LLM functionality or production readiness.

Use the resume drafts in the [interview guide](interview-guide.md) only after reviewing them against your actual work. Disclose AI coding assistance honestly and be ready to explain or change the code yourself.
