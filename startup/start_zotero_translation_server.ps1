param(
  [string]$ContainerName = "translation-server",
  [int]$Port = 1969
)

$ErrorActionPreference = "Stop"

function Require-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "未找到命令: $name"
  }
}

Require-Command docker

$existing = docker ps --filter "name=^/$ContainerName$" --format "{{.Names}}"
if ($existing -contains $ContainerName) {
  Write-Output "translation-server 已在运行: http://127.0.0.1:$Port"
  exit 0
}

$stopped = docker ps -a --filter "name=^/$ContainerName$" --format "{{.Names}}"
if ($stopped -contains $ContainerName) {
  docker rm -f $ContainerName | Out-Null
}

Write-Output "拉取 zotero/translation-server Docker 镜像..."
docker pull zotero/translation-server

Write-Output "启动 translation-server..."
docker run -d -p "${Port}:1969" --rm --name $ContainerName zotero/translation-server | Out-Null

Write-Output "translation-server 已启动: http://127.0.0.1:$Port"
