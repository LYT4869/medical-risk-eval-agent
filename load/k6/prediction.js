import http from 'k6/http';
import {check, sleep} from 'k6';
import {Counter, Rate} from 'k6/metrics';
import {authenticatedHeaders} from './auth.js';

const rejected = new Counter('treesem_503_rejections');
const unexpected = new Rate('treesem_unexpected_responses');
export const options = {
  scenarios: {
    warmup: {executor: 'constant-vus', vus: 1, duration: '30s'},
    steady_4: {executor: 'constant-vus', vus: 4, duration: '60s', startTime: '30s'},
    steady_16: {executor: 'constant-vus', vus: 16, duration: '60s', startTime: '90s'},
    burst_64: {executor: 'constant-vus', vus: 64, duration: '60s', startTime: '150s'},
  },
  thresholds: {treesem_unexpected_responses: ['rate<0.01']},
};
const base = __ENV.TREESEM_LOAD_BASE_URL || 'http://127.0.0.1:8080';
export default function () {
  const response = http.post(`${base}/api/v1/predictions`, JSON.stringify({sample_index: __ITER % 100}), {
    headers: authenticatedHeaders(base),
  });
  if (response.status === 503) rejected.add(1);
  if (response.status === 503) sleep(0.05);
  unexpected.add(response.status !== 200 && response.status !== 503);
  check(response, {'prediction succeeds or sheds load': r => r.status === 200 || r.status === 503});
}
