FROM nginx:alpine

# Copy toàn bộ file giao diện vào thư mục chứa trang web của Nginx
COPY app/frontend /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]