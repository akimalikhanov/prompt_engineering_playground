#!/bin/sh
set -eu

echo "[init] waiting for MinIO to become available…"
# Try setting alias until it succeeds
until /usr/bin/mc alias set minio http://minio:9000 "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null 2>&1; do
  echo "[init] …still waiting"
  sleep 2
done

echo "[init] connected to MinIO, ensuring bucket '${MINIO_BUCKET}' exists…"
# If ls succeeds, bucket exists; otherwise create it
if /usr/bin/mc ls "minio/${MINIO_BUCKET}" >/dev/null 2>&1; then
  echo "[init] bucket '${MINIO_BUCKET}' already exists, skipping"
else
  /usr/bin/mc mb "minio/${MINIO_BUCKET}"
  echo "[init] bucket '${MINIO_BUCKET}' created"
fi

# (Optional) enable versioning for MLflow artifacts
# /usr/bin/mc version enable "minio/${MINIO_BUCKET}" || true

echo "[init] MinIO initialization complete ✅"
