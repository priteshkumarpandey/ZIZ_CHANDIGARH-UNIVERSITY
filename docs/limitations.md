# Limitations and Safety Disclaimers

## Clinical Decision Support Disclaimer
GlucoTwin is strictly a clinical decision support system (CDSS) for licensed physicians and healthcare professionals. It is not an autonomous diagnostic device, nor does it prescribe medications or automate insulin delivery.

## Known Limitations
1. **Synthetic Data Priors**: Although synthetic generators are calibrated against validated literature parameters (AI-READI, CGMacros, and Indian dietary patterns), individual physiological responses in clinical practice can vary due to unmeasured factors like severe acute illness, systemic infection, or corticosteroid therapy.
2. **Sensor Calibration Artifacts**: Continuous glucose monitors (CGMs) measure interstitial fluid glucose, which exhibits a physiological lag of 5 to 15 minutes compared to venous blood glucose during rapid glycemic transitions.
3. **South Asian Heterogeneity**: Regional dietary habits across India encompass diverse culinary traditions. While standard reference literature provides raw ingredient parameters, complex home-cooked regional dishes (such as dosa, idli, and biryani) use illustrative team estimates and may have different glycemic responses.
4. **Physiology Model Assumptions**: The Bergman minimal model assumes simplified two-compartment kinetics. Extreme physical exertion or severe renal impairment require individualized physician assessment.
