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
