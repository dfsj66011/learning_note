# [Dissecting BERT Appendix: The Decoder](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)

[![Miguel Romero Calvo](https://miro.medium.com/fit/c/96/96/1*3hQBk5ZsFUiC0O1k_kCRSw.jpeg)](https://medium.com/@mromerocalvo)

[Miguel Romero Calvo](https://medium.com/@mromerocalvo)Follow

[Nov 27, 2018](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f) · 11 min read

<img src="https://miro.medium.com/max/1400/1*HQhbF6EOR6cL_ZAiHSkqmA.jpeg" width="300">

Unleash the force of self-attention

*This is the appendinx of Understanding BERT written by Miguel Romero and Francisco Ingham. Each article was written jointly by both authors. If you did not already, please refer to* [*Part 1*](https://medium.com/@mromerocalvo/6dcf5360b07f) *to understand the Encoder architecture in depth since this post assumes its prior understanding.*

*Many thanks to Yannet Interian for her revision and feedback.*

# Notation

Before we begin, let’s define the notation we will use throughout the article:

> **emb_dim**: Dimension of the embedding
>
> **input_length**: Length of the input sequence
>
> **target_length**: Length of the target sequence + 1. The +1 is a consequence of the shift.
>
> **vocab_size**: Amount of words in your vocabulary (deviated from the corpus).
>
> **target input**: We will use this term interchangeably to describe the input string (set of sentences) or sequence in the decoder.

# Introduction

The **Transformer** is an attention-based architecture for Natural Language Processing (NLP) that was introduced in the paper **Attention Is All You Need** a year ago.

In this blog post, we are going to examine the Decoder in depth; the part of the Transformer architecture that are not used in [BERT](https://arxiv.org/pdf/1810.04805.pdf). We will reference the [Encoder](https://medium.com/dissecting-bert/encoder-architecture-d3c3d495cdb3) to explain the full **Transformer**’s architecture.

> **Note:** If you are only interested in understanding how BERT works, the parts of the **Transformer** described in this blog post are not relevant.

This article is organized as follows:

1. The Problem the **Transformer** aims to solve.
2. The Information Flow.
3. The Decoder.

# Problem

The problem that the **Transformer** addresses is translation. To translate a sentence into another language, we want our model to:

- Be able to capture the relationships between the words in the input sentence.
- Combine the information contained in the input sentence and what has already been translated at each step.

Imagine that the goal is to translate a sentence from English to Spanish and we are given the following sequences of tokens:

> X = [‘Hello’, ‘,’, ‘how’, ‘are’, ‘you’, ‘?’] (Input sequence)
> Y = [‘Hola’, ‘,’, ‘como’, ‘estas’, ‘?’] (Target sequence)

First, we want to process the information in the input sequence *X* by combining the information in each of the words of the sequence. This is done inside the Encoder.

Once we have this information in the output of the Encoder we want to combine it with the target sequence. This is done in the Decoder.

Encoder and Decoder are specific parts of the **Transformer** architecture as illustrated in Figure 1. We will investigate the Decoder in detail in *Level 3-b: Layers*.

<img src="https://miro.medium.com/max/1400/1*Ekte5vEsR54slWTy6wrhoA.png" width="350">

Figure 1: **Transformer** architecture divided into Encoder and Decoder

# Information Flow

The data flow through the architecture is as follows:

1. The model represents each token as a vector of dimension *emb_dim*. We then have a matrix of dimensions *(input_length) x (emb_dimb)* for a specific input sequence.
2. It then adds positional information (positional encoding). This step will return a matrix of dimensions *(input_length) x (emb_dim)*, just as in the previous step.
3. The data goes through N encoder blocks. After that, we obtain a matrix of dimensions *(input_length) x (emb_dim)*.
4. The target sequence is masked and sent through the decoder’s equivalent of 1) and 2). The output has dimensions *(target_length) x (emb_dim)*.
5. The result of 4) goes through N decoder blocks. In each of the iterations, the decoder is using the encoder’s output 3). This is represented in Figure 2 by the arrows from the encoder to the decoder. The dimensions of the output are *(target_length) x (emb_dim)*.
6. Finally, it applies a fully connected layer and a row-wise softmax. The dimensions of the output are *(target_length) x (vocab_size).*

<img src="https://miro.medium.com/max/1400/1*P2cHPZmuKncBhNmybAlynQ.png" width="300">

The dimensions of the input and the output of the decoder blocks are the same. Hence, it makes sense to use the output of one decoder block as the input of the next decoder block.

**Note**: In the Attention Is All You Need experiments, N was chosen to be 6.

> Note: The blocks do not share weights

# Inputs

Remember that the described algorithm is processing both the input sentence and the target sentence to train the network. The input sentence will be encoded as described in [The Encoder’s architecture](https://medium.com/dissecting-bert/encoder-architecture-d3c3d495cdb3). In this section, we discuss how given a target sentence (e.g. “Hola, como estás ?”) we obtain a matrix representing the target sentence for the decoder blocks.

The process is exactly the same. It is also composed of two general steps:

1. Token embeddings
2. Encoding of the positions.

The main difference is that the target sentence is shifted. That is, before padding, the target sequence will be as follows:

> [“Hola”, “, “, “como”, “estás”, “?”]→[“<SS>”, “Hola”, “, “, “como”, “estás”, “?”]

The rest of the process to vectorize the target sequence will be exactly as the one described for input sentences in [The Encoder’s architecture](https://medium.com/dissecting-bert/encoder-architecture-d3c3d495cdb3).

# Decoder

In this section, we will cover those parts of the decoder that differ from those covered in the encoder.

# Decoder block — Training vs Testing

<img src="https://miro.medium.com/max/712/1*3VAGzPTi7auz_oBv7-Xf9g.png" width="200">

During test time we don’t have the ground truth. The steps, in this case, will be as follows:

1. Compute the embedding representation of the input sequence.
2. Use a starting sequence token, for example ‘\<SS\>’ as the first target sequence: [\<SS\>]. The model gives as output the next token.
3. Add the last predicted token to the target sequence and use it to generate a new prediction [‘\<SS\>’, Prediction_1,…,Prediction_n]
4. Do step 3 until the predicted token is the one representing the End of the Sequence, for example \<EOS\>.

During training we have the ground truth, i.e. the tokens we would like the model to output for every iteration of the above process. Since we have the target in advance, we will give the model the whole shifted target sequence at once and ask it to predict the non-shifted target.

Following up with our previous examples we would input:

> [‘\<SS\>’,’Hola’, ‘,’, ‘ como’, ‘estas’, ‘?’]

and the expected prediction would be:

> [’Hola’, ‘,’, ‘ como’, ‘estas’, ‘?’,’\<EOS\>’]

However, there is a problem here. What if the model sees the expected token and uses it to predict itself? For example, it might see *‘estas’* at the right of *‘como’* and use it to predict *‘estas’*. That’s not what we want because the model will not be able to do that a testing time.

We need to modify some of the attention layers to prevent the model of seeing information on the right (or down in the matrix of vector representation) but allow it to use the already predicted words.

Let’s illustrate this with an example. Given:

> [‘<SS>’,’Hola’, ‘,’, ‘ como’, ‘estas’, ‘?’]

we will transform it into a matrix as described above and add positional encoding. This would result in a matrix:

<img src="https://miro.medium.com/max/1400/1*sYbUKTi-k6o5smZlBbzU5w.png" width="300">

And just as in the encoder the output of the decoder block will be also a matrix of sizes *(target_length) x (emb_dim)*. After a row-wise linear (a linear layer in the form of matrix product on the right) and a Softmax per row this will result in a matrix for which the maximum element per row indicates the next word.

That means that the row assigned to “\<SS\>” is in charge of predicting “Hola”, the row assigned to “Hola” is in charge of predicting “,” and so on. Hence, to predict “estas” we will allow that row to directly interact with the green region but not with the red region in Figure 13.

<img src="https://miro.medium.com/max/1400/1*a4V1WXN-CDwoJE5vyVR4NQ.png" width="400">

Observe that we don’t have problems in the linear layers because they are defined to be token-wise/row-wise in the form of a matrix multiplication through the right.

The problem will be in **Multi-Head Attention** and the input will need to be masked. We will talk more about masking in the next section.

At training time, the prediction of all rows matter. Given that at prediction time we are doing an iterative process we are just going to care about the prediction of the next word of the last token in the target/output sequence.

## Masked Multi-Head Attention

<img src="https://miro.medium.com/max/760/1*B_pAfVZEU-7np0Y7LBrZdw.png" width="200">

This will work exactly as the **Multi-Head Attention** mechanism but adding masking to our input.

The only **Multi-Head Attention** block where masking is required is the first one of each decoder block. This is because the one in the middle is used to combine information between the encoded inputs and the outputs inherited from the previous layers. There is no problem in combining every target token’s representation with any of the input token’s representations (since we will have all of them at test time).

The modification will take place after computing:

<img src="https://miro.medium.com/max/404/1*VGKAJY_w1KKUYn4L0eGdkA.png" width="50">

Observe that this is a matrix such as:

<img src="https://miro.medium.com/max/1400/1*GysQ0m1lL4awcymVpCxc6Q.png" width="400">

Now, the masking step is just going to set to minus infinity all the entries in the strictly upper triangular part of the matrix. In our example:

<img src="https://miro.medium.com/max/1400/1*qWPzSPMytnv5zS89cciFkA.png" width="350">

That’s it! The rest of the process is identical as described in the **Multi-Head Attention** for the encoder.

Let’s now dig in what does it means mathematically to set those elements to minus infinity. Observe that if those entries are relative attention measures per each row, the larger they are, the more attention we need to pay to that token. So setting those elements to minus infinity is mainly saying: “ For the row assigned to predicting *“estás”* (the one with input *“como”)*, ignore *“estás”* and “*?*”. Our Softmax output would look like this:

<img src="https://miro.medium.com/max/1400/1*Xu2X2zrGfTVAp9rvjVq4aA.png" width="500">

The relative attention of those tokens that we were trying to ignore has indeed gone to zero.

When multiplying this matrix with *V_i* the only elements that will be accounted for to predict the next word are the ones into its right, i.e. the ones that the model will have access to during test time.

Observe that this time the output of the modified **Multi-Head Attention**layer will be a matrix of dimensions *(target_length) x (emb_dim)* because the sequence from which it has been calculated has a sequence length of *target_length.*

# Multi-Head Attention — Encoder output and target

<img src="https://miro.medium.com/max/616/1*TTpBMsvq_8tmttJji1TWSg.png" width="150">

Observe that in this case we are using different inputs for that layer. More specifically, instead of deriving *Q_i*, *K_i* and *V_i* from *X* as we have been doing in previous **Multi-Head Attention** layers, this layer will use both the Encoder’s final output *E* (final result of all encoder blocks) and the Decoder’s previous layer output *D* (the masked **Multi-Head Attention** after going through the **Dropout,** **Add & Norm** layer).

Let’s first clarify the shape of those inputs and what they represent:

1. *E*, the encoded input sequence, is a matrix of dimensions *(input_length) x (emb_dim)* which has encoded, by going through 6 encoder blocks, the relationships between the input tokens.
2. *D*, the output from the masked **Multi-Head Attention** after going through the Add & Norm, is a matrix of dimensions *(target_length) x (emb_dim)*.

Let’s now dive into what to do with those matrices. We will use weighted matrices with the same dimensions as before:

<img src="https://miro.medium.com/max/768/1*UILj3J57TLdYcySUvSRX4w.png" width="200">

But this time the projection generating *Q_i* will be done using *D* (target information), while the ones generating *K* and *V* will be created using *E*(input information).

<img src="https://miro.medium.com/max/1096/1*8j2ylRN4IfJV56NYUKSdog.png" width="300">

for every head i=1,…, h.

The matrix *W_0* used after the concatenation of the heads will have dimensions *(d_v\*h) x (emb_dim)* just like the one used in the encoder block.

Apart from that, the **Multi-Head Attention** block is exactly the same as the one in the encoder.

Observe that in this case, the matrix resulting from:

<img src="https://miro.medium.com/max/676/1*T4MCL9SdNaQUC5PVLr-PVQ.png" width="150">

is describing relevant relationships between “encoded input tokens” and “encoded target tokens”. Moreover notice that this matrix will have dimensions *(target_length)x(input_length)*.

Following up on our example:

<img src="https://miro.medium.com/max/1012/1*O9OkEHqFK6ffKtdgsX7R3g.png" width="300">

As we can see, every projected row-token in the target attends to all the positions in the (encoded) input sequence. This matrix encodes the relationship between the input sequence and the target sequence.

Repeating the notation we used in the **Multi-Head Attention** of the Encoder the multiplication with $V_i$ results in:

<img src="https://miro.medium.com/max/1284/1*k8kVGcHvKy3_ItXzYyWItQ.png" width="400">

<img src="https://miro.medium.com/max/1400/1*61XBHADgigA6igSXTF93-Q.png" width="400">

As we can see, every token in the target sequence is represented in every head as a combination of encoded input tokens. Moreover, this will happen for multiple heads and just as before, that is going to allow each token of the target sequence to be represented by multiple relationships with the tokens in the input sequence.

Just like in the encoder block, once the concatenation has been carried out, we will take the product of that with $W_0$

<img src="https://miro.medium.com/max/532/1*K50k7y_ThfTudF-8FUR5Bg.png" width="200">

# Linear and Softmax

<img src="https://miro.medium.com/max/592/1*ccrz7fNllR2mmIc5lSeGSw.png" width="200">

This is the final step before being able to get the predicted token for every position in the target sequence. If you are familiar with Language Models, this is identical to their last layers. The output from the last Add & Norm layer of the last Decoder block is a matrix X of dimensions *(target_length)x(emb_dim).*

The idea of the linear layer is for every row in *x* of *X* to compute:

<img src="https://miro.medium.com/max/200/1*BJU_slrWn-h1_0q9oU3Yhw.png" width="50">

where *W_1* is a matrix of learned weights of dimensions *(emb_dim) x (vocab_size)*. Therefore, for a specific row the result will be a vector of length *vocab_size.*

Finally, a softmax is applied to this vector resulting in a vector describing the probability of the next token. Therefore, taking the position corresponding to the maximum probability returns the most likely next word according to the model.

In matrix form this looks like:

<img src="https://miro.medium.com/max/140/1*2ozBz8_LLxku9uk5ZPTVig.png" width="50">

And applying a Softmax in each resulting row.

*If you have any feedback please let us know in the comment section!*

# References

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf); Vaswani et al., 2017.

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf); Devlin et al., 2018.

[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax); Alexander Rush, Vincent Nguyen and Guillaume Klein.

[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf); Howard et al., 2018.

[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf); Radford et al., 2018.