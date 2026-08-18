import http from 'k6/http';
import {check} from 'k6';
import {Rate, Trend} from 'k6/metrics';
import {authenticatedHeaders} from './auth.js';

const unexpected = new Rate('treesem_resilience_unexpected');
const healthLatency = new Trend('treesem_health_latency', true);
export const options = {
  scenarios: {burst: {executor: 'constant-vus', vus: 64, duration: '60s'}},
  thresholds: {
    treesem_resilience_unexpected: ['rate<0.01'],
    treesem_health_latency: ['p(95)<250'],
  },
};

const base = __ENV.TREESEM_LOAD_BASE_URL || 'http://127.0.0.1:8080';
export default function () {
  const prediction = http.post(`${base}/api/v1/predictions`,
    JSON.stringify({sample_index: __ITER % 100}), {
      headers: authenticatedHeaders(base), timeout: '10s'});
  unexpected.add(prediction.status !== 200 && prediction.status !== 503);
  const health = http.get(`${base}/health`, {timeout: '1s'});
  healthLatency.add(health.timings.duration);
  check(health, {'health remains responsive under burst': r => r.status === 200});
}
