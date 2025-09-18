# DEFT
Emergent Properties of Efficient Fine-Tuning in Text-to-Image Models

![Emergent Properties of Efficient Fine-Tuning in Text-to-Image Models](assets/teaser.png)

## Overview
In this work, we introduce DEFT, Decompositional Efficient Fine-Tuning, an efficient fine-tuning
framework that adapts a pre-trained weight matrix by decomposing its update into
two components with two trainable matrices: (1) a projection onto the complement
of a low-rank subspace spanned by a low-rank matrix, and (2) a low-rank update.
The single trainable low-rank matrix defines the subspace, while the other trainable
low-rank matrix enables flexible parameter adaptation within that subspace. We
conducted extensive experiments on the Dreambooth and Dreambench Plus datasets
for personalization, the InsDet dataset for object and scene adaptation, and the
VisualCloze dataset for a universal image generation framework through visual
in-context learning with both Stable Diffusion and a unified model. Our results
demonstrated compatative performance and highlights the emergent properties of
efficient fine-tuning.

## Quick strart
### [./docs/inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources)
### create an enviroment please follow the [./docs/enviroment.md#enviroment-details](doc)