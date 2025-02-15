# Hugging Face tool-augmented example

focusing on building a lightweight agent that integrates tools like search and retrieval. This aligns with GIC Lab’s research themes (e.g., tool-augmented systems, efficiency) while being beginner-friendly. Here’s the plan:

#### Step 1: Set Up Environment

Goal: Install dependencies and configure tools.

```
!pip install transformers[agents] smolagents langchain sentence-transformers faiss-cpu duckduckgo-search
```





#### Step 2: Build a Simple Tool-Augmented Agent

Objective: Create an agent that uses a search tool and a retriever tool.

```
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchToolfrom langchain_community.vectorstores import FAISSfrom langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize tools

search_tool = DuckDuckGoSearchTool() # Built-in search tool :cite[3]embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5") # Lightweight embedding model :cite[9]

# Create a retriever tool (simplified version)

class RetrieverTool: name = "retriever" description = "Retrieves documents from a knowledge base." def __init__(self, docs): self.vectordb = FAISS.from_documents(docs, embedding_model) def forward(self, query: str) -> str: return "\n".join([doc.page_content for doc in self.vectordb.similarity_search(query, k=3)])

# Load sample documents (e.g., from Hugging Face datasets)

from datasets import load_datasetdataset = load_dataset("m-ric/huggingface_doc", split="train"):cite[3]retriever_tool = RetrieverTool(dataset)

# Initialize the agent with tools

model = HfApiModel("gpt2") # Use GPT-2-small for low compute :cite[6]agent = CodeAgent(tools=[search_tool, retriever_tool], model=model)

# Run the agent

response = agent.run("What is the transformers library used for? Search the web and cross-reference with the knowledge base.")print(response)
```



#### Step 3: Test and Optimize

Key Actions:

Benchmark Latency: Use %%timeit to measure inference speed on your RTX 4070.

Validate Outputs: Compare answers from the search tool vs. the retriever tool.

Troubleshoot:

If CUDA memory errors occur, reduce batch size or use FP16:

```
model = HfApiModel("gpt2", device_map="auto", torch_dtype=torch.float16)
```

For larger models, use quantization:

```
from transformers import BitsAndBytesConfigquantization_config = BitsAndBytesConfig(load_in_4bit=True)model = HfApiModel("meta-llama/Llama-3-8B-Instruct", quantization_config=quantization_config)
```



Step 4: Visualize Results
Example Output Analysis:

Tool Used    Response Snippet    Latency (s)
Search    "Transformers is a library for NLP tasks like text generation..."    1.2
Retriever    "Hugging Face Transformers provides APIs for BERT, GPT-2..."    0.8
Visualization (using Matplotlib):




```
import matplotlib.pyplot as pltlabels = ['Search Tool', 'Retriever Tool']latencies = [1.2, 0.8]plt.bar(labels, latencies, color=['blue', 'green'])plt.title('Tool Performance Comparison')plt.ylabel('Latency (seconds)')plt.show()
```



Step 5: Connect to GIC Lab’s Research
Efficiency Focus:

- Compare your tool-augmented agent’s token usage with vanilla LLMs (use transformers’ tokenizer).

- Experiment with pruning/quantization to reduce compute costs 6.

Reliability:

- Test adversarial queries (e.g., ambiguous questions) to evaluate robustness 6.

Documentation:

- Write a 1-page report highlighting how this aligns with GIC Lab’s work on ReWOO (decoupled reasoning) and Adaptix (efficient inference) 13.

Troubleshooting Common Issues

- Tool Dependency Errors: Ensure all tools are initialized with correct parameters (e.g., RetrieverTool requires pre-embedded documents) 3.

- API Limits: Use local models (e.g., GPT-2) instead of Hugging Face API if hitting rate limits.

- GPU Memory: Use smaller models (e.g., T5-small) or gradient checkpointing 9.

Next Steps

1. Expand the toolkit with a calculator or image generator (see smolagents’ multimodal examples) 3.

2. Publish your code on GitHub and share it with your advisor as a proof-of-concept!


