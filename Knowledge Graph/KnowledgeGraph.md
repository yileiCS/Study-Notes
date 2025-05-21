# [Knowledge Graph (KG)](https://www.ibm.com/think/topics/knowledge-graph)

Resources:
[Introduction to Knowledge Graph](https://www.icourse163.org/course/ZJU-1464119172)

## What is knowledge graph?
- Known as a **Semantic Network**
- A structured representation of knowledge using nodes (entities) and edges (relationships)
- Directed-Labled Graph
- represents a network of real-world entities
  - such as: objects, events, situations or concepts
  - illustrates the relationship between them
- This information is usually stored in a graph database and visualized as **graph structure**
- consists of: 
  - a set of **nodes**
  - a set of **edges** connecting the nodes 
  - a set of **labels/labels**  
  - an assignment function which associates an edge with a label
- Any object, place, or person can be a node. An edge defines the relationship between the nodes

## Technical Essence 技术内涵
### Graph-based Knowledge Representation 基于图的知识表示
**RDF**
- Resource Description Framework (RDF)
- An RDF triple (S, P, O) encodes a statement - a simple **logical expression**, or claim about the world

### Graph Data Storage and Query 图数据存储与查询
- Graph databases fully utilize graph structure to establish micro-indexing 图数据库充分利用图的结构建立微索引
  - Micro-indexing is more cost-effective than global indexing in relational databases when processing traversal queries 微索引相比关系数据库的全局索引在处理遍历查询时更加廉价
  - Query complexity is independent of the overall dataset size, only proportional to the size of the adjacent subgraph 查询复杂度与数据集整体大小无关，仅正比于相邻子图的大小

### Knowledge Base Population
- Knowledge extraction from data with different sources and structures to form knowledge stored in knowledge graphs 从不同来源、不同结构的数据中进行知识提取，形成知识存入到知识图谱
- Text is generally not used as the initial source for knowledge graph construction, but rather for knowledge graph completion 文本一般不作为知识图谱构建的初始来源，而用作知识图谱补全

### Knowledge Graph Integration 知识图谱融合

### Knowledge Graph Reasoning 知识图谱推理
- Symbol logic-based reasoning methods: OWL Reasoners, Datalog, Rete, etc. 基于符号逻辑的推理办法
- Reasoning methods based on graph structure or representation learning: PRA, AMIE, TransE, Analogy, DeepPath, NeuralLP 基于图结构或表示学习的推理方法

### Knowledge Graph Question Answering 知识图谱问答 KBQA

### Graph Algorithms and Graph Neural Networks 图算法与图神经网络


---
## Ontology
- A formal and structured representation of knowledge, including concepts, relationships, and properties within a specific domain 
- defines the structure and relationships of concepts within a specific domain
- serve to create a formal representation of the entities in the KG
- purpose: provides a consistent and well-defined structure for organizing and representing data 
- usually based on a **taxanomy**
  - can contain multiple taxonomies, it maintains its own separate definition
- has similar manner with KGs: i.e. nodes and edges, and are based on the **Resource Description Framework (RDF) triples**

---
## How a knowledge graph works?
- KGs are typically made up of datasets from various sources, which frequently differ in structure.
- **Schemas**, **identities** and **context** work together to provide structure to diverse data.

### Fueled by machine learning
KGs, that are fueled by machine learning, utilize **natural language processing (NLP)** to construct a comprehensive view of nodes, edges, and labels through a process called **semantic enrichment**. 
- When data is ingested, this process allows KGs to:
  - identify individual objects
  - understand the relationships between different objects
- Once a KG is complete, it allows question answering and search systems to retrieve and reuse comprehensive answers to given queries

---
## Applications of KGs
- Search engines
- Smart question answering and recommendation systems
- Big data analytics: knowledge graph can normalize and fuse data from multiple sources, and through graph reasoning, support complex association data mining 知识图谱通过规范化语义融合多来源数据，并能通过图谱推理能力支持复杂关联数据的挖掘分析 

### KGs-related Projects:
- [YAGO](https://yago-knowledge.org/getting-started)
- [DBPedia](https://www.dbpedia.org/resources/knowledge-graphs/)
- [Wikidata](https://www.wikidata.org/wiki/Q33002955)
- [OpenKG.CN](http://onegraph.openkg.cn/)
- [WordNet](https://wordnet.princeton.edu/)
- [ConceptNet](https://conceptnet.io/)
- [BabelNet](https://babelnet.org/)
- Autonomous Driving Network


Knowledge Graph Embedding (KGE) remains the most widely adopted method for representing knowledge in knowledge graphs and integrating it into machine learning architectures 知识图谱嵌入（KGE）仍然是表示知识图谱中的知识并将其集成到机器学习架构中最广泛采用的方法
