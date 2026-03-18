# Volumetric Radar Echo Motion Estimation Using Physics-Informed Deep Learning: A Case Study Over Slovakia
Repository containing the code needed to replicate the volumetric motion field extimator from the paper [*Volumetric Radar Echo Motion Estimation Using Physics-Informed Deep Learning: A Case Study Over Slovakia*](https://arxiv.org/abs/2603.13589).

### Abstract

In precipitation nowcasting, most extrapolation-based methods rely on two-dimensional radar composites to estimate the horizontal motion of precipitation systems. However, in some cases, precipitation systems can exhibit varying motion at different heights. We propose a physics-informed convolutional neural network that estimates independent horizontal motion fields for multiple altitude layers directly from volumetric radar reflectivity data and investigate the practical benefits of altitude-wise motion field estimation for precipitation nowcasting. The model is trained end-to-end on volumetric observations from the Slovak radar network and its extrapolation nowcasting performance is evaluated. We compare the proposed model against an architecturally identical baseline operating on vertically pooled two-dimensional radar composites. Our results show that, although the model successfully learns altitude-wise motion fields, the estimated displacement is highly correlated across vertical levels for the vast majority of precipitation events. Consequently, the volumetric approach does not yield systematic improvements in nowcasting accuracy. While categorical metrics indicate increased precipitation detection at longer lead times, this gain is largely attributable to non-physical artifacts and is accompanied by a growing positive bias. A comprehensive inter-altitude motion field correlation analysis further confirms that events exhibiting meaningful vertical variability in horizontal motion are rare in the studied region. We conclude that, for the Slovak radar dataset, the additional complexity of three-dimensional motion field estimation is not justified by questionable gains in predictive skill. Nonetheless, the proposed framework remains applicable in climates where precipitation systems exhibit stronger vertical variability in horizontal motion.

**A more detailed README with instructions for replication to be added.**

## Links
### Dataset archive: [https://drive.google.com/file/d/1jjX21crezHQtJEPDncYguEmCQN_h6cN6](https://drive.google.com/file/d/1jjX21crezHQtJEPDncYguEmCQN_h6cN6)
- Data from the Slovak radar network - four dual-pol doppler radars
- roughly 3.5 years
- quantized, zipped and containing only the volumetric reflectivity fields, the size is 13 GB

### Dataset metadata file: [https://drive.google.com/file/d/14wdZWo0wUcV_cEaSlHG_HY1TT9HZowxr](https://drive.google.com/file/d/14wdZWo0wUcV_cEaSlHG_HY1TT9HZowxr)
- needed for the dataloader to work
