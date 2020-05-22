# Gender-bias Idea Applied to Political Tendency

## Background
- [Kaiwei's Group](https://github.com/PatriciaXiao/TwitterText/blob/master/reference/ML-seminar-201910.pdf) has many previous works investigating Gender Bias.
- Our problem is different from theirs, as our text context is conditioned, instead of that the text itself is biased or not.
  - But, assuming that we want to learn about people's biased opinions towards the two parties, what should be do?
  - First, learn a embedding (or having a pretrained embedding)
  - Then, use some anchor word to define the hyperplane that points out the boundary of the two parties

## Working in Progress
- Implemented [our own GloVe model](./glove_forced_polarity.py) in pytorch
- Use Kaiwei objective (TODO)
  - Their objective is different from ours: they categorized **ALL** vocabulary into three groups: neutral, female, male; and have some male-female word pairs.
- Please be sure to run on the latest pyTorch, they have had some important fixations that we rely on

phrase-mining group
- simple / robust 
- dataset : phrases / words

left => -1
right => 1
J_D^{L_2} => scaling problem
beta => hyper-parameter

vg => reasonable to stay stable

scaling => average / lambda

J_G => not normalize in Kaiwei paper

make everything comparible

choose the hyper-parameters on validation set

1D -> 2D (random y)

evaluation:
1. case-study 
2. sentence-level embedding: task (classification on w^(g))
3. predicting the **link** (w, separate) - retweet means more than follow

