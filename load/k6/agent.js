import http from 'k6/http';
import exec from 'k6/execution';
import {check, sleep} from 'k6';
import {Rate} from 'k6/metrics';

const unexpected = new Rate('treesem_agent_unexpected_responses');
export const options = {
  scenarios: {
    agent_1: {executor: 'constant-vus', vus: 1, duration: '60s'},
    agent_4: {executor: 'constant-vus', vus: 4, duration: '60s', startTime: '60s'},
    agent_16: {executor: 'constant-vus', vus: 16, duration: '60s', startTime: '120s'},
    agent_64_burst: {executor: 'constant-vus', vus: 64, duration: '60s', startTime: '180s'},
  },
  thresholds: {treesem_agent_unexpected_responses: ['rate<0.01']},
};

const base = __ENV.TREESEM_LOAD_BASE_URL || 'http://127.0.0.1:8080';
const token = __ENV.TREESEM_LOAD_ACCESS_TOKEN || '';
export default function () {
  const question = (__ITER % 2 === 0)
    ? '解释一下当前预测结果'
    : '介绍产后出血的权威资料并给出引用';
  const response = http.post(`${base}/api/v1/chat`, JSON.stringify({message: question}), {
    headers: {
      'Content-Type': 'application/json',
      'Idempotency-Key': `k6-${exec.vu.idInTest}-${exec.scenario.iterationInTest}`,
      ...(token ? {Authorization: `Bearer ${token}`} : {}),
    },
    timeout: '35s',
  });
  unexpected.add(![200, 409, 503, 504].includes(response.status));
  check(response, {'agent completes or sheds load safely': r => [200, 409, 503, 504].includes(r.status)});
  sleep(0.05);
}
