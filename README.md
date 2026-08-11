# Transformer (from scratch)

Originally introduced in the 2017 paper "Attention is All You Need" (Vaswani et al., 2017), transformers form the structure for most modern large language models, including ChatGPT and Claude. In fact, the GPT in ChatGPT stands for Generative Pre-trained *Transformer*. 

Transformer architecture blends small feed-forward MLPs with attention mechanisms and has been shown "to be superior in quality while being more parallelizable and requiring significantly less time to train" () when compared to previous model designs. 

## Full Notebook
![Full Notebook](./screenshots/Full.png)

## Step 0: Setup
The transformer that I will be creating is very similar to the one described in "Attention is All You Need" (). It will be used for translating a short English sentence, "The dog ran fast." into German. 

Computers do not understand written language the way humans do, but they do understand numbers. Therefore, we need to translate the initial sentence into a numeric representation. We can do this by breaking up the sentence into different pieces or tokens. In this case, I am just setting everything to lowercase, removing the punctuation, and using the whitespace to determine where to start the next token. This tokenization method results in each token being a word, but there are other methods like subword tokenization that may work better at scale. 

The tokens are then turned into vector embeddings

Looking at the Hyperparameters/Structure, there are 4 initial tokens, and 4 numbers that make up the vector embedding per token (dmodel). As we will see later, each attention block will have 2 heads, and each feed-forward block will have a hidden layer with 8 neurons. I decided to use 2 encoder and 2 decoder layers to show how these layers stack and interact; however, I only did the math for the first layer to avoid repetition. Since this is a translation task, we need the initial vocabulary and the vocabulary of the language we are translating to. In this case, the English vocab size is 4, and the German's is 6. That means there is just enough for the German translation and for the <start> and <end> tags.

![Step 0](./screenshots/Step0.png)

The architecture on the right side of Step 0 is a representation of the original figure from "Attention is All You Need" and will be used throughout this example to show where we are in the process. 

<img src="./screenshots/Transformer_Architecture.png" width="500">

**Orginal figure from "Attention is All You Need"*


## Step 1: Positional Encoding

![Step 1](./screenshots/Step1.png)

The raw embedding now contains learnable parameters that represent the semantic meaning of each token, but they do not have any context for where the tokens are positioned in the sequence. For this, we need to add positional encoding. The most obvious way to do this would be to have a sequence from 1 to n tokens and just add that to the embedding. However, if there is a 1,000-word sequence, then you would be completely overriding the original embedding signal with massive positional encoded values. Instead, we can use sinusoidal positional encoding, which is also used in the paper. This method uses sine and cosine pairs; each set of pairs changes at different rates, which act similarly to the hands of a clock. If you have 2 pairs, you can think of these as the minute and hour hands of a clock, where the first pair changes at a much faster rate than the second. So while position 10 might look similar to position 110 on the first pair, the added context of the lower-frequency second pair is enough to differentiate them. This method allows us to add positional context within the bounds of the raw embeddings to retain signal from both. The formula used to create these encodings in the paper is shown below:

![Positional Encoding](./screenshots/Positional_Encoding.png)

For this example, we will have 2 pairs, or "hands," which add up to the dmodel of 4. We will call these frequency 0 and 1, as seen in Part 1. These frequencies have denominators of 1 and 100, respectively. With the frequency denominators calculated, we can now find the sine and cosine values for each token position, as seen in parts 2-5. Then those pairs are concatenated and stacked into a matrix. As you can see from the first pair in Part 6, the values vary significantly from row to row, yet the second pair changes much more gradually between rows. This relationship is shown on the graph below:

![Positional Encoding Graph](./screenshots/Positional_Encoding_Graph.png)

As you can see from the graph, the difference between tokens 2 and 3 is clearly visible for frequency 0, and the dots for tokens 2 and 3 practically overlap for frequency 1.

## Step 2:
![Step 2](./screenshots/Step2.0.png)

![Step 2.1](./screenshots/Step2.1.png)

## Step 3:
![Step 3](./screenshots/Step3.png)

## Step 4-6:
![Step 4](./screenshots/Step4.0.png)

![Step 4.1](./screenshots/Step4.1.png)

## Step 7:
![Step 7](./screenshots/Step7.png)

## Step 8-9:
![Step 8](./screenshots/Step8.png)

## Step 10:
![Step 10](./screenshots/Step10.png)

## Step 11-13:
![Step 11](./screenshots/Step11.png)

## Step 14-15:
![Step 14](./screenshots/Step14.png)

## Step 16:
![Step 16](./screenshots/Step16.png)
