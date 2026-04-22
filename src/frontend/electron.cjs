const { app, BrowserWindow } = require('electron')
const { spawn, execSync }    = require('child_process')
const path                   = require('path')
const http                   = require('http')

const API_PORT   = 8000
const API_URL    = `http://127.0.0.1:${API_PORT}`
const VITE_URL   = 'http://localhost:5173'
const isDev      = !app.isPackaged

let mainWindow = null
let apiProcess = null


// ── Find a working Python executable ──────────────────────────────────────
function findPython() {
  const candidates = process.platform === 'win32'
    ? ['python', 'python3', 'py']
    : ['python3', 'python']

  for (const cmd of candidates) {
    try {
      execSync(`${cmd} --version`, { stdio: 'ignore', shell: true })
      console.log(`[Electron] Using Python: ${cmd}`)
      return cmd
    } catch {
      // not found, try next
    }
  }
  throw new Error('No Python executable found. Install Python and ensure it is on PATH.')
}


// ── Kill API process ───────────────────────────────────────────────────────
function killApi() {
  if (!apiProcess) return
  if (process.platform === 'win32') {
    try {
      execSync(`taskkill /pid ${apiProcess.pid} /T /F`, { stdio: 'ignore' })
    } catch {}
  } else {
    apiProcess.kill()
  }
  apiProcess = null
}


// ── Start Python FastAPI ───────────────────────────────────────────────────
function startApi() {
  const backendDir = isDev
    ? path.join(__dirname, '..', 'backend')
    : path.join(process.resourcesPath, 'backend')

  if (isDev) {
    // Dev mode — use Python directly
    let python
    try {
      python = findPython()
    } catch (e) {
      console.error('[Electron]', e.message)
      return
    }

    apiProcess = spawn(python, ['api.py'], {
      cwd: backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: true,
    })
  } else {
    // Production — use bundled exe
    const apiExe = path.join(backendDir, 'jaginator_api.exe')
    console.log(`[Electron] Starting API from: ${apiExe}`)

    apiProcess = spawn(apiExe, [], {
      cwd: backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
    })
  }

  apiProcess.stdout.on('data', d => console.log('[API]', d.toString().trim()))
  apiProcess.stderr.on('data', d => console.error('[API ERR]', d.toString().trim()))
  apiProcess.on('exit', code => console.log(`[API] exited with code ${code}`))
}


// ── Wait for API to be ready ───────────────────────────────────────────────
function waitForApi(retries = 30) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      http.get(`${API_URL}/health`, res => {
        if (res.statusCode === 200) resolve()
        else if (n > 0) setTimeout(() => attempt(n - 1), 1000)
        else reject(new Error('API did not start in time'))
      }).on('error', () => {
        if (n > 0) setTimeout(() => attempt(n - 1), 1000)
        else reject(new Error('API did not start in time'))
      })
    }
    attempt(retries)
  })
}


// ── Create window ──────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width:  1280,
    height: 800,
    minWidth:  900,
    minHeight: 600,
    title: 'Jaginator',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  if (isDev) {
    mainWindow.loadURL(VITE_URL)
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'))
  }

  mainWindow.on('closed', () => { mainWindow = null })
}


// ── App lifecycle ──────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  startApi()
  try {
    await waitForApi()
    console.log('[Electron] API ready')
  } catch (e) {
    console.error('[Electron] API failed to start:', e.message)
  }
  createWindow()
})

app.on('before-quit', killApi)
app.on('will-quit', killApi)

app.on('window-all-closed', () => {
  killApi()
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (mainWindow === null) createWindow()
})