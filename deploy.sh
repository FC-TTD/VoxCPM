#!/bin/bash
set -e

CI_MODE=false
CD_MODE=false
SYNC_ONLY=false
API_MODE=false
ANSIBLE_DIR="./ansible"

while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI_MODE=true
      shift
      ;;
    --cd)
      CD_MODE=true
      shift
      ;;
    sync)
      SYNC_ONLY=true
      shift
      ;;
    docker)
      TAGS="docker"
      shift
      ;;
    api)
      TAGS="api"
      shift
      ;;
    *)
      echo "Usage: $0 [--ci] [--cd] docker | api | sync"
      echo "Examples:"
      echo "  $0 docker         # 部署 Demo/Train (Docker Compose)"
      echo "  $0 api            # 部署 API (Swarm Stack via CI/CD role)"
      echo "  $0 --ci api       # 仅构建 API 镜像"
      echo "  $0 --cd api       # 仅部署 API Stack"
      exit 1
      ;;
  esac
done

if [ "$SYNC_ONLY" = true ]; then
  echo "仅同步..."
  ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "sync"
  exit 0
fi

if [ -z "$TAGS" ]; then
    echo "Error: 请指定 docker 或 api"
    exit 1
fi

# Append ci/cd tags if specified
if [ "$CI_MODE" = true ]; then
    TAGS="$TAGS,ci"
fi
if [ "$CD_MODE" = true ]; then
    TAGS="$TAGS,cd"
fi

# Check for problematic tag combinations
if [ "$TAGS" = "docker,cd" ]; then
    echo "错误：docker 和 cd 标签不能同时使用"
    echo "请使用以下替代方案："
    echo "  ./deploy.sh docker     # 仅 Docker Compose 部署"
    echo "  ./deploy.sh api        # API Swarm Stack 部署"
    echo "  ./deploy.sh --cd api   # 仅部署 API Stack"
    exit 1
fi

echo "Running playbook with tags: $TAGS"
ansible-playbook -i "$ANSIBLE_DIR/inventory.yml" "$ANSIBLE_DIR/site.yml" --tags "$TAGS"
