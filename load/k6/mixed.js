import http from 'k6/http';
import {check, sleep} from 'k6';
import {authenticatedHeaders} from './auth.js';

export const options = {vus: 4, duration: '60s', thresholds: {
  http_req_failed: ['rate<0.05'], http_req_duration: ['p(95)<1500']}};
const base = __ENV.TREESEM_LOAD_BASE_URL || 'http://127.0.0.1:8080';
export default function () {
  const headers = authenticatedHeaders(base);
  const prediction = http.post(`${base}/api/v1/predictions`, JSON.stringify({sample_index: __ITER % 100}), {headers});
  check(prediction, {'prediction 200': r => r.status === 200});
  if (prediction.status === 200) {
    const id = prediction.json('prediction_id');
    check(http.get(`${base}/api/v1/predictions/${id}/explanation`, {headers}), {'explanation 200': r => r.status === 200});
    check(http.get(`${base}/api/v1/sessions/current/history?limit=20`, {headers}), {'history 200': r => r.status === 200});
  }
  sleep(0.05);
}
