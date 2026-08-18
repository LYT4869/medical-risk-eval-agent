FROM nginx:1.27.4-alpine
COPY deploy/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/web /usr/share/nginx/html
RUN rm -f /usr/share/nginx/html/nginx.conf
EXPOSE 8080
