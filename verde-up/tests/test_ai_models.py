import os
import tempfile
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
from PIL import Image


class AiModelFallbackTests(unittest.TestCase):
    def setUp(self):
        rotation = patch.object(app, '_GEMINI_MODEL_ROTATION_INDEX', 0)
        rotation.start()
        self.addCleanup(rotation.stop)

    @staticmethod
    def create_image():
        image_buffer = BytesIO()
        image = Image.new('RGB', (100, 80), (180, 180, 180))
        image.save(image_buffer, format='PNG')
        image.close()
        image_buffer.seek(0)
        return image_buffer

    def test_primary_success_skips_backup_and_model_discovery(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(return_value=SimpleNamespace(text='resultado'))
            )
        )

        with patch.object(app, 'MODELOS_GEMINI', ('modelo-principal', 'modelo-respaldo')):
            result = app.generar_con_respaldo(client, None, 'prompt')

        self.assertEqual(result, 'resultado')
        client.models.generate_content.assert_called_once()
        self.assertEqual(
            client.models.generate_content.call_args.kwargs['model'],
            'modelo-principal'
        )
        config = client.models.generate_content.call_args.kwargs['config']
        self.assertEqual(config.thinking_config.thinking_budget, 0)
        self.assertEqual(config.max_output_tokens, 480)

    def test_request_start_model_rotates_between_configured_models(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=[
                        SimpleNamespace(text='primer resultado'),
                        SimpleNamespace(text='segundo resultado')
                    ]
                )
            )
        )

        with patch.object(app, 'MODELOS_GEMINI', ('gemini-3.8-flash', 'gemini-3.5-flash-lite')):
            app.generar_con_respaldo(client, None, 'prompt')
            app.generar_con_respaldo(client, None, 'prompt')

        self.assertEqual(
            [
                call.kwargs['model']
                for call in client.models.generate_content.call_args_list
            ],
            ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
        )

    def test_failed_primary_uses_only_one_backup(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=[
                        TimeoutError('principal agotó el tiempo'),
                        SimpleNamespace(text='resultado de respaldo')
                    ]
                )
            )
        )

        with patch.object(app, 'MODELOS_GEMINI', ('modelo-principal', 'modelo-respaldo')):
            result = app.generar_con_respaldo(client, None, 'prompt')

        self.assertEqual(result, 'resultado de respaldo')
        self.assertEqual(
            [
                call.kwargs['model']
                for call in client.models.generate_content.call_args_list
            ],
            ['modelo-principal', 'modelo-respaldo']
        )

    def test_failure_after_backup_stops_and_preserves_timeout(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=[
                        TimeoutError('principal agotó el tiempo'),
                        RuntimeError('respaldo no disponible')
                    ]
                )
            )
        )

        with patch.object(app, 'MODELOS_GEMINI', ('modelo-principal', 'modelo-respaldo')):
            with self.assertRaises(app.GeminiTimeoutError):
                app.generar_con_respaldo(client, None, 'prompt')

        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_client_is_reused_and_has_timeout_without_sdk_retries(self):
        with patch.object(app, '_GEMINI_CLIENT', None):
            with patch.object(app, '_GEMINI_CLIENT_API_KEY', None):
                with patch.object(app.genai, 'Client') as client_factory:
                    first = app.crear_cliente_gemini('api-key')
                    second = app.crear_cliente_gemini('api-key')

        self.assertIs(first, second)
        client_factory.assert_called_once()
        options = client_factory.call_args.kwargs['http_options']
        self.assertEqual(options.timeout, app.GEMINI_TIMEOUT_MS)
        self.assertEqual(options.retry_options.attempts, 1)

    def test_sdk_timeout_is_classified_and_wrapped(self):
        self.assertTrue(app.es_timeout_gemini(app.httpx2.ReadTimeout('agotado')))
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=[
                        app.httpx2.ReadTimeout('principal agotado'),
                        app.httpx2.ReadTimeout('respaldo agotado')
                    ]
                )
            )
        )

        with patch.object(app, 'MODELOS_GEMINI', ('modelo-principal', 'modelo-respaldo')):
            with self.assertRaises(app.GeminiTimeoutError):
                app.generar_con_respaldo(client, None, 'prompt')

    def test_timeout_on_primary_is_reported_if_backup_also_fails(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=Mock(
                    side_effect=[
                        app.httpx2.ReadTimeout('principal agotado'),
                        RuntimeError('respaldo no disponible')
                    ]
                )
            )
        )

        with patch.object(app, 'MODELOS_GEMINI', ('modelo-principal', 'modelo-respaldo')):
            with self.assertRaises(app.GeminiTimeoutError):
                app.generar_con_respaldo(client, None, 'prompt')

    def test_scan_route_uses_local_fallback_on_gemini_timeout(self):
        with tempfile.TemporaryDirectory() as upload_folder:
            original_upload_folder = app.app.config['UPLOAD_FOLDER']
            app.app.config['UPLOAD_FOLDER'] = upload_folder
            try:
                image = self.create_image()
                with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
                    with patch.object(app, 'crear_cliente_gemini'):
                        with patch.object(
                            app,
                            'generar_con_respaldo',
                            side_effect=app.GeminiTimeoutError('tiempo agotado')
                        ):
                            response = app.app.test_client().post(
                                '/api/analizar',
                                data={'imagen': (image, 'planta.png')},
                                content_type='multipart/form-data'
                            )
            finally:
                app.app.config['UPLOAD_FOLDER'] = original_upload_folder

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['modo'], 'local')
        self.assertIn('Diagnóstico local orientativo', response.get_json()['resultado'])
        self.assertNotIn('error', response.get_json())

    def test_scan_route_uses_local_fallback_on_provider_failure(self):
        with tempfile.TemporaryDirectory() as upload_folder:
            original_upload_folder = app.app.config['UPLOAD_FOLDER']
            app.app.config['UPLOAD_FOLDER'] = upload_folder
            try:
                with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
                    with patch.object(app, 'crear_cliente_gemini'):
                        with patch.object(
                            app,
                            'generar_con_respaldo',
                            side_effect=RuntimeError('cuota agotada')
                        ):
                            response = app.app.test_client().post(
                                '/api/analizar',
                                data={'imagen': (self.create_image(), 'planta.png')},
                                content_type='multipart/form-data'
                            )
            finally:
                app.app.config['UPLOAD_FOLDER'] = original_upload_folder

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['modo'], 'local')
        self.assertNotIn('cuota agotada', response.get_data(as_text=True))

    def test_missing_api_key_uses_local_image_analysis(self):
        with tempfile.TemporaryDirectory() as upload_folder:
            original_upload_folder = app.app.config['UPLOAD_FOLDER']
            app.app.config['UPLOAD_FOLDER'] = upload_folder
            try:
                with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
                    response = app.app.test_client().post(
                        '/api/analizar',
                        data={'imagen': (self.create_image(), 'planta.png'), 'tipo': 'espacio'},
                        content_type='multipart/form-data'
                    )
            finally:
                app.app.config['UPLOAD_FOLDER'] = original_upload_folder

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['modo'], 'local')
        self.assertIn('180/255', response.get_json()['resultado'])

    def test_image_compression_stays_within_target_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, 'noisy.png')
            image = Image.frombytes('RGB', (800, 800), os.urandom(800 * 800 * 3))
            image.save(image_path, format='PNG')
            image.close()

            part, luminosidad = app.preparar_imagen_ligera(image_path)

        self.assertLessEqual(len(part.inline_data.data), app.GEMINI_IMAGE_MAX_BYTES)
        self.assertGreater(luminosidad, 100)
        self.assertLess(luminosidad, 155)

    def test_ai_response_includes_timing_header_and_request_id(self):
        response = app.app.test_client().post(
            '/api/guia-personalizada',
            json={'materiales': ''}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('X-AI-Request-ID', response.headers)
        self.assertIn('input_parse;dur=', response.headers['Server-Timing'])
        self.assertIn('app_total;dur=', response.headers['Server-Timing'])


if __name__ == '__main__':
    unittest.main()
