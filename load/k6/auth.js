import http from 'k6/http';
import {fail, sleep} from 'k6';

let cachedHeaders = null;

export function authenticatedHeaders(base) {
  if (cachedHeaders !== null) return cachedHeaders;
  const accessToken = __ENV.TREESEM_LOAD_ACCESS_TOKEN || '';
  const sessionCookie = __ENV.TREESEM_LOAD_SESSION_COOKIE || '';
  if (accessToken && sessionCookie) {
    cachedHeaders = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
      Cookie: `treeSemSession=${sessionCookie}`,
    };
    return cachedHeaders;
  }
  const email = __ENV.TREESEM_LOAD_EMAIL || '';
  const password = __ENV.TREESEM_LOAD_PASSWORD || '';
  if (!email || !password) {
    fail('set TREESEM_LOAD_EMAIL and TREESEM_LOAD_PASSWORD, or provide both access token and session cookie');
  }
  let response = null;
  for (let attempt = 0; attempt < 12; attempt += 1) {
    response = http.post(`${base}/api/v1/auth/login`, JSON.stringify({email, password}), {
      headers: {'Content-Type': 'application/json'},
      tags: {operation: 'load_login'},
    });
    if (response.status === 200) break;
    if (![429, 503].includes(response.status)) fail(`load login failed with HTTP ${response.status}`);
    sleep(Math.min(0.05 * (2 ** attempt), 0.5));
  }
  if (response === null || response.status !== 200) fail('load login remained overloaded after bounded retries');
  cachedHeaders = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${response.json('access_token')}`,
  };
  return cachedHeaders;
}
