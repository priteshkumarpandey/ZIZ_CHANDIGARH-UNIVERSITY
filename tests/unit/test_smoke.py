import unittest


class TestModuleSmoke(unittest.TestCase):
    def test_import_config_modules(self):
        from src.glucotwin.config.settings import settings

        self.assertIsNotNone(settings.PROJECT_NAME)
        self.assertEqual(settings.PROJECT_NAME, "GlucoTwin")

    def test_import_data_modules(self):
        import src.glucotwin.data.adapters.ai_readi_synthetic as ai_readi_synthetic
        import src.glucotwin.data.adapters.cgmacros as cgmacros
        import src.glucotwin.data.adapters.hupa_ucm as hupa_ucm
        import src.glucotwin.data.schema as schema

        self.assertIsNotNone(schema.PatientsSchema)
        self.assertIsNotNone(cgmacros.CGMacrosAdapter)
        self.assertIsNotNone(ai_readi_synthetic.AIREADISyntheticAdapter)
        self.assertIsNotNone(hupa_ucm.HUPAAdapter)

    def test_import_features_modules(self):
        import src.glucotwin.features.builder as builder
        import src.glucotwin.features.windowing as windowing

        self.assertIsNotNone(builder.FeatureWindowBuilder)
        self.assertIsNotNone(windowing.WindowedDataset)

    def test_import_synth_modules(self):
        import src.glucotwin.synth.meals as meals
        import src.glucotwin.synth.personas as personas

        self.assertGreater(len(meals.INDIAN_MEAL_CATALOG), 0)
        self.assertGreater(len(meals.IFCT_FOOD_CATALOG), 0)
        self.assertGreater(len(personas.SYNTHETIC_PERSONAS), 0)

    def test_import_twin_modules(self):
        import src.glucotwin.twin.simulator as simulator
        import src.glucotwin.twin.state as state

        self.assertIsNotNone(state.TwinStateManager)
        self.assertIsNotNone(simulator.PhysiologySimulator)

    def test_import_summary_modules(self):
        import src.glucotwin.summary.generator as generator

        self.assertIsNotNone(generator.ClinicalSummaryGenerator)

    def test_import_serving_modules(self):
        import src.glucotwin.serving.app as app
        import src.glucotwin.serving.schemas.fhir as fhir
        import src.glucotwin.serving.schemas.requests as requests
        import src.glucotwin.serving.schemas.responses as responses

        self.assertIsNotNone(fhir.PatientProfile)
        self.assertIsNotNone(requests.PredictRequest)
        self.assertIsNotNone(responses.ForecastResponse)
        self.assertIsNotNone(app.app)

    def test_import_streaming_modules(self):
        import src.glucotwin.streaming.kafka as kafka
        import src.glucotwin.streaming.replay as replay

        self.assertIsNotNone(kafka.kafka_manager)
        self.assertIsNotNone(replay.TelemetryReplaySimulator)

    def test_import_utils_modules(self):
        import src.glucotwin.utils.io as io_util
        import src.glucotwin.utils.logging as logging_util
        import src.glucotwin.utils.seeds as seeds

        self.assertIsNotNone(logging_util.setup_logging)
        self.assertIsNotNone(logging_util.logger)
        self.assertIsNotNone(seeds.set_global_seed)
        self.assertIsNotNone(io_util.read_parquet)
