from fastembed import SparseTextEmbedding

documents = "You should stay, study and sprint."


model = SparseTextEmbedding(model_name="Qdrant/bm25")
embeddings = list(model.passage_embed(documents))
print(embeddings)

# [
#     SparseEmbedding(
#         values=array([1.67419738, 1.67419738, 1.67419738, 1.67419738]),
#         indices=array([171321964, 1881538586, 150760872, 1932363795])),
#     SparseEmbedding(values=array(
#         [1.66973021, 1.66973021, 1.66973021, 1.66973021, 1.66973021]),
#                     indices=array([
#                         578407224, 1849833631, 1008800696, 2090661150,
#                         1117393019
#                     ]))
# ]
