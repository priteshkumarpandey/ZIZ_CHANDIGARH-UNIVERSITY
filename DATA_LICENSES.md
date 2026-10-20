# Data Licenses and Attributions

This document records the licensing terms, attributions, and redistribution status for every dataset and external reference utilized or referenced by the GlucoTwin project.

## Summary Table

| Data Source | License / Terms | Raw Data Redistributed? | Source Citation / Access Reference |
| :--- | :--- | :---: | :--- |
| **CGMacros** | CC BY-NC-SA 4.0 (checked on the PhysioNet page) | **No** | Gutierrez-Osuna, R., Kerr, D., Mortazavi, B., & Das, A. (2025). CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. https://doi.org/10.13026/3z8q-x658.<br><br>PhysioNet reference: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215-e220. |
| **HUPA-UCM** | License CC BY 4.0, seen on the institutional repository record, confirmed on the Mendeley Data page | **No** | Citation: Hidalgo JI, Alvarado J, Botella M, Aramendi A, Velasco JM, Garnica O. HUPA-UCM diabetes dataset. Data in Brief 55 (2024) 110559, doi 10.1016/j.dib.2024.110559; dataset doi 10.17632/3hbcscwz44.1. |
| **Synthetic AI-READI (ADVANCE Center, Vanderbilt University Medical Center and Washington University)** | Terms not published on the pages reviewed, raw files not redistributed | **No** | Source https://hiplab.vumc.org/synthetix, derived from the NIH Bridge2AI AI-READI cohort, cite 'A synthetic multi-modal dataset for type 2 diabetes', Scientific Reports. |
| **Synthea** | Not used yet | **No** | Synthetic patient cohort generator framework. |
| **IFCT 2017** | Raw ingredient nutritional values used with acknowledgment, no tables redistributed | **No** | Longvah, T., Ananthan, R., Bhaskarachary, K., & Venkaiah, K. (2017). Indian Food Composition Tables. National Institute of Nutrition, ICMR, Hyderabad. Raw ingredient proximate and available carbohydrate values were extracted locally from the reference book to compute cooked dish nutrients. No tables or book text are redistributed in this repository. Dish nutrient values in `data/external/indian_meal_templates.csv` are computed from extracted ingredient nutrients and recipe assumptions in `data/external/meal_recipes.csv`. Meal recipes, cooking adjustments, and glycemic index values are team assumptions. |

## Important Redistribution and Non-Commercial Policy

1. **No Raw Data Redistribution**: No raw data from any source listed above is redistributed in this repository. All raw clinical files, sensor recordings, and external database dumps must be acquired directly from their respective repositories by authorized users.
2. **Derivative Artifacts Licensing**: Because the machine learning models, scalers, calibrators, and operating thresholds were trained on the CGMacros dataset (licensed under CC BY-NC-SA 4.0), all model checkpoints in `models/checkpoints/` and derived evaluation reports in `reports/` are shared under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0).
3. **No Unverified License Claims**: No license is stated or claimed as verified for any external source unless it has been explicitly confirmed (CGMacros on PhysioNet).
