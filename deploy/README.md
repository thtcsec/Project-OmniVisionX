# Deploy (Docker overrides)

Tất cả file liên quan **chỉ nằm trong thư mục `deploy/` ở root repo** — không đặt ra ngoài dự án.

## Bind mount Postgres / Redis (Windows hoặc cần đường cố định)

Mặc định `docker-compose.yml` dùng **named volume**. Nếu muốn dữ liệu nằm trên ổ đĩa (ví dụ backup dễ):

1. Tạo thư mục (một lần), từ **root repo**:

   ```bash
   mkdir -p deploy/postgres-data deploy/redis-data
   ```

2. Copy file ví dụ ra **cùng thư mục với `docker-compose.yml`** (root repo):

   ```bash
   copy deploy\docker-compose.override.example.yml docker-compose.override.yml
   ```

   (Linux/macOS: `cp deploy/docker-compose.override.example.yml docker-compose.override.yml`)

3. `docker compose up -d` — Compose sẽ **merge** `docker-compose.override.yml` tự động.

Đường mount trong override dùng **`./deploy/postgres-data`** và **`./deploy/redis-data`** (relative tới root repo), không trỏ ra ngoài cây dự án.

## Git

Thư mục `deploy/postgres-data/` và `deploy/redis-data/` đã **gitignore** (dữ liệu DB không commit).
