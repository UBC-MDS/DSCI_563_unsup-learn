import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy import spatial
torch.manual_seed(1)
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt 

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import sys
sys.path.append("code/.")
from preprocessing import MyPreprocessor

from sklearn.metrics.pairwise import cosine_similarity

class skipgramModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=10, context_size=2):
        super(skipgramModel, self).__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight)        
        self.linear = nn.Linear(embedding_dim, vocab_size)
        nn.init.uniform_(self.linear.weight)
        # self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view((1, -1))
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
def create_input_pairs(pp_corpus, word2idx, context_size=2):
    idx_pairs = [] 
    for sentence in pp_corpus:
        indices = [word2idx[word] for word in sentence]
        for center_word_pos in range(len(indices)):
            for w in range(-context_size, context_size + 1):
                context_word_pos = center_word_pos + w
                if (
                    context_word_pos < 0
                    or context_word_pos >= len(indices)
                    or center_word_pos == context_word_pos
                ):
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs
    )  # it will be useful to have this as numpy array
    return idx_pairs

def get_vocab(tokenized_corpus):
    vocab = []
    for sent in tokenized_corpus:
        for token in sent:
            if token not in vocab:
                vocab.append(token)

    return vocab     
        
def train_model(model, idx_pairs, n_epochs=10):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(n_epochs):
        total_loss = 0
        for inp, target in idx_pairs:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            word_idx = torch.tensor(inp, dtype=torch.long)
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(word_idx)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    # print(losses)  # The loss decreased every iteration over the training data!    
    
    
def plot_word2_vec_pca(
    data,
    words,
    show_labels=False,
    size=100,
    title="PCA visualization",
):
    """
    Carry out dimensionality reduction using PCA and plot 2-dimensional clusters.

    Parameters
    -----------
    data : numpy array
        data as a numpy array
    words : list
        the original raw sentences for labeling datapoints
    show_labels : boolean
        whether you want to show labels for points or not (default: False)
    size : int
        size of points in the scatterplot
    title : str
        title for the visualization plot

    Returns
    -----------
    None. Shows the clusters.
    """

    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_comp, columns=["pca1", "pca2"])

    plt.figure(figsize=(10, 7))
    plt.title(title)
    ax = sns.scatterplot(
        x="pca1", y="pca2", data=pca_df, palette="tab10", s=size
    )

    x = pca_df["pca1"].tolist()
    y = pca_df["pca2"].tolist()
    if show_labels:
        for i, txt in enumerate(words):
            plt.annotate(txt, (x[i]+0.05, y[i]+0.05), size=15)

    plt.show()   
    

def plot_embeddings(n_epochs, model, idx_pairs, vocab, word2idx, word1="hockey", word2="football", word3="mango"):
    train_model(model, idx_pairs, n_epochs)
    emb_mat = model.embedding.weight.detach().numpy()
    context_mat = model.linear.weight.detach().numpy()    
    print("Number of epochs: ", n_epochs)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Target embeddings", "Context embeddings")
    )    
    fig.add_trace(go.Heatmap(z=emb_mat, y=vocab, colorscale="Viridis"), row=1, col=1)

    fig.add_trace(go.Heatmap(z=context_mat, y=vocab, colorscale="Viridis"), row=1, col=2)

    fig.update_layout(height=700, showlegend=False)
    
    plot_word2_vec_pca(emb_mat, vocab, show_labels=True)
    #sim1 = round(1 - spatial.distance.cosine(emb_mat[word2idx[word1]], emb_mat[word2idx[word2]]), 4)
    #sim2 = round(1 - spatial.distance.cosine(emb_mat[word2idx[word1]], emb_mat[word2idx[word3]]), 4)    
    vec1 = emb_mat[word2idx[word1]].reshape(1, -1)
    vec2 = emb_mat[word2idx[word2]].reshape(1, -1)    
    vec3 = emb_mat[word2idx[word3]].reshape(1, -1)    
    sim1 = cosine_similarity(vec1, vec2)[0][0]
    sim2 = cosine_similarity(vec1, vec3)[0][0]
    print(f"Similarity between {word1} and {word2} is {sim1}")    
    print(f"Similarity between {word1} and {word3} is {sim2}")        
    fig.show()
        
if __name__=="__main__":
    toy_corpus = [
    "drink mango juice",
    "drink pineapple juice",
    "drink apple juice",
    "drink squeezed pineapple juice", 
    "drink squeezed mango juice",     
    "drink apple tea",
    "drink mango tea",
    "drink mango water",
    "drink apple water",
    "drink pineapple water",
    "drink juice",
    "drink water",
    "drink tea",
    "play hockey",
    "play football",
    "play piano",
    "piano play",    
    "play hockey game",
    "play football game" ]
    
    
    EMBEDDING_DIM = 10
    CONTEXT_SIZE = 2
    toy_pp_corpus = MyPreprocessor(toy_corpus)    
    vocab = get_vocab(toy_pp_corpus)    
    word2idx = {w: idx for (idx, w) in enumerate(vocab)}
    idx2word = {idx: w for (idx, w) in enumerate(vocab)}
    print(vocab)

    idx_pairs = create_input_pairs(toy_pp_corpus, word2idx)    
    model = skipgramModel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    print('The number of input pairs: ', len(idx_pairs))
    train_model(model, idx_pairs, n_epochs=10)
