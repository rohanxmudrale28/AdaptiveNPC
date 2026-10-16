
## Iteration: 0: baseline env.yaml (PLAN defaults)

```
Win-rate matrix (row vs col, draws = 1/2, both sides):
                    rusher    camper  repeater     mixer  switcher     kiter counter_p scripted_    random
rusher               0.500     0.143     0.945     0.990     0.640     0.277     0.000     0.362     0.998
camper               0.857     0.500     0.540     0.609     0.620     0.737     0.252     0.620     0.648
repeater             0.055     0.460     0.500     0.559     0.397     0.009     0.000     0.001     0.598
mixer                0.010     0.391     0.441     0.500     0.329     0.007     0.003     0.002     0.522
switcher             0.360     0.380     0.603     0.671     0.500     0.269     0.058     0.248     0.688
kiter                0.723     0.263     0.991     0.993     0.731     0.500     0.001     0.492     1.000
counter_puncher      1.000     0.748     1.000     0.997     0.942     0.999     0.500     0.677     0.970
scripted_npc         0.638     0.380     0.999     0.998     0.752     0.508     0.323     0.500     1.000
random               0.003     0.352     0.402     0.478     0.312     0.000     0.030     0.000     0.500

Draw-rate matrix (unordered pair):
                    rusher    camper  repeater     mixer  switcher     kiter counter_p scripted_    random
rusher               0.633     0.065     0.064     0.004     0.207     0.427     0.000     0.706     0.000
camper               0.065     0.925     0.898     0.717     0.656     0.079     0.132     0.068     0.667
repeater             0.064     0.898     1.000     0.752     0.695     0.005     0.000     0.003     0.692
mixer                0.004     0.717     0.752     0.702     0.540     0.003     0.003     0.002     0.728
switcher             0.207     0.656     0.695     0.540     0.802     0.148     0.043     0.190     0.532
kiter                0.427     0.079     0.005     0.003     0.148     0.662     0.000     0.549     0.001
counter_puncher      0.000     0.132     0.000     0.003     0.043     0.000     0.403     0.000     0.052
scripted_npc         0.706     0.068     0.003     0.002     0.190     0.549     0.000     1.000     0.000
random               0.000     0.667     0.692     0.728     0.532     0.001     0.052     0.000     0.780

Pairs with draw rate > 50%: rusher vs rusher (63.3%), rusher vs scripted_npc (70.6%), camper vs camper (92.5%), camper vs repeater (89.8%), camper vs mixer (71.7%), camper vs switcher (65.6%), camper vs random (66.8%), repeater vs repeater (100.0%), repeater vs mixer (75.2%), repeater vs switcher (69.5%), repeater vs random (69.2%), mixer vs mixer (70.2%), mixer vs switcher (54.0%), mixer vs random (72.8%), switcher vs switcher (80.2%), switcher vs random (53.2%), kiter vs kiter (66.2%), kiter vs scripted_npc (54.9%), scripted_npc vs scripted_npc (100.0%), random vs random (78.0%)

Checks:
  [PASS] mirror matches in [0.45, 0.55]: all ok
  [FAIL] no dominant archetype (>= 0.65 vs all others): counter_puncher
  [FAIL] every non-random archetype wins >=1 and loses >=1 matchup: no win: -; no loss: ['counter_puncher']
  [FAIL] every non-random archetype has a counter >= 0.60: counter_puncher (best counter scripted_npc 0.323)
OVERALL: FAIL
```

## Iteration: my label

```
Win-rate matrix (row vs col, draws = 1/2, both sides):
                    rusher    camper  repeater     mixer  switcher     kiter counter_p scripted_    random
rusher               0.500     0.143     0.945     0.990     0.640     0.277     0.000     0.362     0.998
camper               0.857     0.500     0.540     0.609     0.620     0.737     0.252     0.620     0.648
repeater             0.055     0.460     0.500     0.559     0.397     0.009     0.000     0.001     0.598
mixer                0.010     0.391     0.441     0.500     0.329     0.007     0.003     0.002     0.522
switcher             0.360     0.380     0.603     0.671     0.500     0.269     0.058     0.248     0.688
kiter                0.723     0.263     0.991     0.993     0.731     0.500     0.001     0.492     1.000
counter_puncher      1.000     0.748     1.000     0.997     0.942     0.999     0.500     0.677     0.970
scripted_npc         0.638     0.380     0.999     0.998     0.752     0.508     0.323     0.500     1.000
random               0.003     0.352     0.402     0.478     0.312     0.000     0.030     0.000     0.500

Draw-rate matrix (unordered pair):
                    rusher    camper  repeater     mixer  switcher     kiter counter_p scripted_    random
rusher               0.633     0.065     0.064     0.004     0.207     0.427     0.000     0.706     0.000
camper               0.065     0.925     0.898     0.717     0.656     0.079     0.132     0.068     0.667
repeater             0.064     0.898     1.000     0.752     0.695     0.005     0.000     0.003     0.692
mixer                0.004     0.717     0.752     0.702     0.540     0.003     0.003     0.002     0.728
switcher             0.207     0.656     0.695     0.540     0.802     0.148     0.043     0.190     0.532
kiter                0.427     0.079     0.005     0.003     0.148     0.662     0.000     0.549     0.001
counter_puncher      0.000     0.132     0.000     0.003     0.043     0.000     0.403     0.000     0.052
scripted_npc         0.706     0.068     0.003     0.002     0.190     0.549     0.000     1.000     0.000
random               0.000     0.667     0.692     0.728     0.532     0.001     0.052     0.000     0.780

Pairs with draw rate > 50%: rusher vs rusher (63.3%), rusher vs scripted_npc (70.6%), camper vs camper (92.5%), camper vs repeater (89.8%), camper vs mixer (71.7%), camper vs switcher (65.6%), camper vs random (66.8%), repeater vs repeater (100.0%), repeater vs mixer (75.2%), repeater vs switcher (69.5%), repeater vs random (69.2%), mixer vs mixer (70.2%), mixer vs switcher (54.0%), mixer vs random (72.8%), switcher vs switcher (80.2%), switcher vs random (53.2%), kiter vs kiter (66.2%), kiter vs scripted_npc (54.9%), scripted_npc vs scripted_npc (100.0%), random vs random (78.0%)

Checks:
  [PASS] mirror matches in [0.45, 0.55]: all ok
  [FAIL] no dominant archetype (>= 0.65 vs all others): counter_puncher
  [FAIL] every non-random archetype wins >=1 and loses >=1 matchup: no win: -; no loss: ['counter_puncher']
  [FAIL] every non-random archetype has a counter >= 0.60: counter_puncher (best counter scripted_npc 0.323)
OVERALL: FAIL
```

## Iteration: 1: flanker added to train; counter_puncher unchanged (noise 0.05, delay 0)

```
Win-rate matrix (row vs col, draws = 1/2, both sides):
                    rusher    camper  repeater     mixer   flanker  switcher     kiter counter_p scripted_    random
rusher               0.500     0.143     0.945     0.990     0.990     0.651     0.297     0.000     0.365     0.994
camper               0.857     0.500     0.540     0.609     0.943     0.621     0.735     0.253     0.635     0.637
repeater             0.055     0.460     0.500     0.559     0.151     0.395     0.007     0.000     0.002     0.596
mixer                0.010     0.391     0.441     0.500     0.005     0.323     0.006     0.002     0.002     0.534
flanker              0.010     0.057     0.849     0.995     0.500     0.471     0.020     0.494     0.021     0.999
switcher             0.349     0.379     0.605     0.677     0.529     0.500     0.273     0.067     0.258     0.685
kiter                0.703     0.265     0.993     0.994     0.980     0.727     0.500     0.000     0.480     0.997
counter_puncher      1.000     0.747     1.000     0.998     0.506     0.933     1.000     0.500     0.679     0.971
scripted_npc         0.635     0.365     0.998     0.998     0.979     0.742     0.520     0.321     0.500     0.999
random               0.006     0.363     0.404     0.466     0.001     0.315     0.003     0.029     0.001     0.500

Decisive win-rate matrix (wins / (wins + losses), row vs col):
                    rusher    camper  repeater     mixer   flanker  switcher     kiter counter_p scripted_    random
rusher               0.500     0.119     0.976     0.992     0.996     0.688     0.141     0.000     0.029     0.994
camper               0.881     0.500     0.893     0.885     0.946     0.852     0.756     0.216     0.645     0.912
repeater             0.024     0.107     0.500     0.737     0.149     0.183     0.005     0.000     0.000     0.811
mixer                0.008     0.115     0.263     0.500     0.005     0.130     0.005     0.001     0.002     0.625
flanker              0.004     0.054     0.851     0.995     0.500     0.471     0.005     0.491     0.003     0.999
switcher             0.312     0.148     0.817     0.870     0.529     0.500     0.233     0.053     0.200     0.899
kiter                0.859     0.244     0.995     0.995     0.995     0.767     0.500     0.000     0.458     0.997
counter_puncher      1.000     0.784     1.000     0.999     0.509     0.947     1.000     0.500     0.679     0.992
scripted_npc         0.971     0.355     1.000     0.998     0.997     0.800     0.542     0.321     0.500     0.999
random               0.006     0.088     0.189     0.375     0.001     0.101     0.003     0.008     0.001     0.500

Draw-rate matrix (unordered pair):
                    rusher    camper  repeater     mixer   flanker  switcher     kiter counter_p scripted_    random
rusher               0.633     0.065     0.064     0.004     0.013     0.198     0.437     0.000     0.714     0.001
camper               0.065     0.925     0.898     0.717     0.007     0.656     0.081     0.129     0.068     0.667
repeater             0.064     0.898     1.000     0.752     0.005     0.668     0.004     0.000     0.003     0.692
mixer                0.004     0.717     0.752     0.702     0.000     0.520     0.002     0.002     0.001     0.729
flanker              0.013     0.007     0.005     0.000     0.303     0.016     0.031     0.315     0.037     0.001
switcher             0.198     0.656     0.668     0.520     0.016     0.787     0.151     0.031     0.193     0.536
kiter                0.437     0.081     0.004     0.002     0.031     0.151     0.657     0.000     0.518     0.001
counter_puncher      0.000     0.129     0.000     0.002     0.315     0.031     0.000     0.427     0.000     0.043
scripted_npc         0.714     0.068     0.003     0.001     0.037     0.193     0.518     0.000     1.000     0.000
random               0.001     0.667     0.692     0.729     0.001     0.536     0.001     0.043     0.000     0.765

Pairs with draw rate > 50%: rusher vs rusher (63.3%), rusher vs scripted_npc (71.4%), camper vs camper (92.5%), camper vs repeater (89.8%), camper vs mixer (71.7%), camper vs switcher (65.6%), camper vs random (66.8%), repeater vs repeater (100.0%), repeater vs mixer (75.2%), repeater vs switcher (66.8%), repeater vs random (69.2%), mixer vs mixer (70.2%), mixer vs switcher (52.0%), mixer vs random (72.9%), switcher vs switcher (78.7%), switcher vs random (53.6%), kiter vs kiter (65.7%), kiter vs scripted_npc (51.8%), scripted_npc vs scripted_npc (100.0%), random vs random (76.5%)

Checks:
  [PASS] mirror matches in [0.45, 0.55]: all ok
  [PASS] no dominant archetype (>= 0.65 vs all others, adjusted or decisive): none dominant
  [FAIL] every non-random archetype wins >=1 and loses >=1 matchup: no win: -; no loss: ['counter_puncher']
  [FAIL] every non-random archetype has a counter >= 0.60: counter_puncher (best counter flanker adj 0.494, decisive 0.491)
OVERALL: FAIL
```

## Iteration: 2: counter_puncher reaction_delay 1, noise 0.2

```
Win-rate matrix (row vs col, draws = 1/2, both sides):
                    rusher    camper  repeater     mixer   flanker  switcher     kiter counter_p scripted_    random
rusher               0.500     0.143     0.945     0.990     0.990     0.651     0.297     0.098     0.365     0.994
camper               0.857     0.500     0.540     0.609     0.943     0.621     0.735     0.304     0.635     0.637
repeater             0.055     0.460     0.500     0.559     0.151     0.395     0.007     0.006     0.002     0.596
mixer                0.010     0.391     0.441     0.500     0.005     0.323     0.006     0.017     0.002     0.534
flanker              0.010     0.057     0.849     0.995     0.500     0.471     0.020     0.636     0.021     0.999
switcher             0.349     0.379     0.605     0.677     0.529     0.500     0.273     0.111     0.258     0.685
kiter                0.703     0.265     0.993     0.994     0.980     0.727     0.500     0.039     0.480     0.997
counter_puncher      0.902     0.696     0.994     0.982     0.364     0.889     0.961     0.500     0.353     0.945
scripted_npc         0.635     0.365     0.998     0.998     0.979     0.742     0.520     0.647     0.500     0.999
random               0.006     0.363     0.404     0.466     0.001     0.315     0.003     0.055     0.001     0.500

Decisive win-rate matrix (wins / (wins + losses), row vs col):
                    rusher    camper  repeater     mixer   flanker  switcher     kiter counter_p scripted_    random
rusher               0.500     0.119     0.976     0.992     0.996     0.688     0.141     0.068     0.029     0.994
camper               0.881     0.500     0.893     0.885     0.946     0.852     0.756     0.284     0.645     0.912
repeater             0.024     0.107     0.500     0.737     0.149     0.183     0.005     0.005     0.000     0.811
mixer                0.008     0.115     0.263     0.500     0.005     0.130     0.005     0.009     0.002     0.625
flanker              0.004     0.054     0.851     0.995     0.500     0.471     0.005     0.661     0.003     0.999
switcher             0.312     0.148     0.817     0.870     0.529     0.500     0.233     0.096     0.200     0.899
kiter                0.859     0.244     0.995     0.995     0.995     0.767     0.500     0.036     0.458     0.997
counter_puncher      0.932     0.716     0.995     0.991     0.339     0.904     0.964     0.500     0.342     0.982
scripted_npc         0.971     0.355     1.000     0.998     0.997     0.800     0.542     0.658     0.500     0.999
random               0.006     0.088     0.189     0.375     0.001     0.101     0.003     0.018     0.001     0.500

Draw-rate matrix (unordered pair):
                    rusher    camper  repeater     mixer   flanker  switcher     kiter counter_p scripted_    random
rusher               0.633     0.065     0.064     0.004     0.013     0.198     0.437     0.069     0.714     0.001
camper               0.065     0.925     0.898     0.717     0.007     0.656     0.081     0.095     0.068     0.667
repeater             0.064     0.898     1.000     0.752     0.005     0.668     0.004     0.002     0.003     0.692
mixer                0.004     0.717     0.752     0.702     0.000     0.520     0.002     0.017     0.001     0.729
flanker              0.013     0.007     0.005     0.000     0.303     0.016     0.031     0.155     0.037     0.001
switcher             0.198     0.656     0.668     0.520     0.016     0.787     0.151     0.037     0.193     0.536
kiter                0.437     0.081     0.004     0.002     0.031     0.151     0.657     0.006     0.518     0.001
counter_puncher      0.069     0.095     0.002     0.017     0.155     0.037     0.006     0.158     0.072     0.077
scripted_npc         0.714     0.068     0.003     0.001     0.037     0.193     0.518     0.072     1.000     0.000
random               0.001     0.667     0.692     0.729     0.001     0.536     0.001     0.077     0.000     0.765

Pairs with draw rate > 50%: rusher vs rusher (63.3%), rusher vs scripted_npc (71.4%), camper vs camper (92.5%), camper vs repeater (89.8%), camper vs mixer (71.7%), camper vs switcher (65.6%), camper vs random (66.8%), repeater vs repeater (100.0%), repeater vs mixer (75.2%), repeater vs switcher (66.8%), repeater vs random (69.2%), mixer vs mixer (70.2%), mixer vs switcher (52.0%), mixer vs random (72.9%), switcher vs switcher (78.7%), switcher vs random (53.6%), kiter vs kiter (65.7%), kiter vs scripted_npc (51.8%), scripted_npc vs scripted_npc (100.0%), random vs random (76.5%)

Checks:
  [PASS] mirror matches in [0.45, 0.55]: all ok
  [PASS] no dominant archetype (>= 0.65 vs all others, adjusted or decisive): none dominant
  [PASS] every non-random archetype wins >=1 and loses >=1 matchup: no win: -; no loss: -
  [PASS] every non-random archetype has a counter >= 0.60: rusher<-counter_puncher adj 0.90/dec 0.93; camper<-counter_puncher adj 0.70/dec 0.72; repeater<-scripted_npc adj 1.00/dec 1.00; mixer<-scripted_npc adj 1.00/dec 1.00; flanker<-rusher adj 0.99/dec 1.00; switcher<-counter_puncher adj 0.89/dec 0.90; kiter<-counter_puncher adj 0.96/dec 0.96; counter_puncher<-scripted_npc adj 0.65/dec 0.66; scripted_npc<-camper adj 0.64/dec 0.64
OVERALL: PASS
```
