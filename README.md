# Results

## Figure 3: Factorizability

![](plots/pythia-1B-fig3.png)
*Pythia-1B*

## Figure 4: Position Intervention Experiments

TODO: explanation

![Swapping positions of A0](plots/fig4_top_v2.png)
*Pythia-1B*

![Swapping positions of A1](plots/fig4_bottom_v2.png)
*Pythia-1B*

## Mean Interventions
Table 1:

![](plots/gemma-2-27b_table2.png.png)
*Gemma-2-27b -- I found that this experiment didn't really work with smaller models -- maybe because they're using a different binding mechanism, or maybe because they're bad at knowing capitals in the first place.*

It seems like kind of a weird choice to use accuract here -- I think the results look worse if you use logits instead (TODO?)