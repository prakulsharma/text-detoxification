# Multilingual Text Detoxification

## Description

This project aims to detoxify multilingual text on social media platforms while preserving the original sentiment and meaning. Inspired by Dementieva et al. (2021), the project uses transformer-based models to convert offensive language into neutral expressions. By experimenting with advanced models like TinyLlama, FLAN-T5, and GreenLlama, I explored their potential in removing toxicity while retaining the core content and sentiment.

## Methods

1. **Baseline Model**: I used Facebook's M2M100 for multilingual translation, combined with BART and T5 for detoxifying English and Russian texts.
2. **FLAN-T5 Integration**: Integrated FLAN-T5, a model adapted from T5 and enhanced for generative tasks.
3. **Fine-Tuning FLAN-T5**: Fine-tuned on the ParaDetox dataset for improved detoxification.
4. **TinyLlama**: Employed this model specifically for English detoxification due to its efficiency.
5. **TinyLlama with LoRA**: Fine-tuned using Low-Rank Adaptation (LoRA) to minimize computational overhead.

## Experiments

The models were assessed using these metrics:

- **Style Transfer Accuracy (STA)**: Evaluates style transformation accuracy.
- **Semantic Similarity (SIM)**: Measures semantic similarity to the original text.
- **CHRF Score**: Character-level n-gram similarity.
- **Joint Metric (J)**: Composite metric combining all three.

### Results

| Model                     | J     | CHRF  | SIM   | STA   |
|---------------------------|-------|-------|-------|-------|
| **Baseline (BART)**       | 0.609 | 0.804 | 0.860 | 0.863 |
| **Flan-T5**               | 0.069 | 0.354 | 0.369 | 0.611 |
| **Tiny Llama**            | 0.015 | 0.083 | 0.296 | 0.494 |
| **Fine-Tuned Flan-T5**    | 0.520 | 0.720 | 0.768 | 0.890 |
| **Fine-Tuned Tiny Llama** | 0.063 | 0.255 | 0.508 | 0.773 |

**Error Analysis**: Issues included lexical substitution errors, awkward phrasing, and inconsistency in detoxifying toxic elements.

## Future Scope

1. **Cross-Lingual Transfer Learning**: Utilize learned representations to boost performance across multiple languages.
2. **Rule-Based Filtering**: Establish rules for identifying toxic content across cultures.
3. **Active Learning**: Optimize the training process by selecting the most informative data samples.

## Conclusion

My work demonstrated the potential of advanced models like FLAN-T5 in multilingual detoxification. However, the baseline BART model remains highly effective. Further exploration of cross-lingual learning and active learning could refine these techniques and contribute significantly to multilingual text detoxification.
