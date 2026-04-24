param(
    [string]$ContainerName = "translation-server",
    [int]$Port = 1969
)

$ErrorActionPreference = "Stop"

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $name"
    }
}

Require-Command docker

$existing = docker ps --filter "name=^/$ContainerName$" --format "{{.Names}}"
if ($existing -contains $ContainerName) {
    Write-Output "translation-server already running: http://127.0.0.1:$Port"
    exit 0
}

$stopped = docker ps -a --filter "name=^/$ContainerName$" --format "{{.Names}}"
if ($stopped -contains $ContainerName) {
    docker rm -f $ContainerName | Out-Null
}

Write-Output "Pulling zotero/translation-server Docker image..."
docker pull zotero/translation-server

Write-Output "Starting translation-server..."
docker run -d -p "${Port}:1969" --rm --name $ContainerName zotero/translation-server | Out-Null

Write-Output "translation-server ready: http://127.0.0.1:$Port"
