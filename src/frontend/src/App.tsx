import { useEffect, useState } from 'react'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Roboto+Condensed:ital,wght@0,300;0,400;0,700;1,400&family=Share+Tech+Mono&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0b0e;
    --bg2:      #10121a;
    --bg3:      #161820;
    --surface:  #1c1e28;
    --gold:     #c9a84c;
    --gold-lt:  #e8c96a;
    --gold-dk:  #8a6820;
    --silver:   #8a9ab0;
    --silver-lt:#b8c8d8;
    --blue:     #1a5aff;
    --blue-lt:  #4a80ff;
    --red:      #cc2200;
    --green:    #00aa55;
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.12);
    --text:     #c8cfd8;
    --text-dim: #5a6070;
    --white:    #f0f2f6;
    --font-head: 'Oswald', sans-serif;
    --font-body: 'Roboto Condensed', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
  }

  html, body, #root {
    min-height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 15px;
    -webkit-font-smoothing: antialiased;
  }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--gold-dk); }

  @keyframes fadein {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fadein { animation: fadein 0.3s ease both; }

  input[type=number]::-webkit-inner-spin-button,
  input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
  input[type=number] { -moz-appearance: textfield; }
`

const TEAM_NAMES = {
  ARI:'Arizona Cardinals',   ATL:'Atlanta Falcons',     BAL:'Baltimore Ravens',
  BUF:'Buffalo Bills',       CAR:'Carolina Panthers',   CHI:'Chicago Bears',
  CIN:'Cincinnati Bengals',  CLE:'Cleveland Browns',    DAL:'Dallas Cowboys',
  DEN:'Denver Broncos',      DET:'Detroit Lions',       GB:'Green Bay Packers',
  HOU:'Houston Texans',      IND:'Indianapolis Colts',  JAX:'Jacksonville Jaguars',
  KC:'Kansas City Chiefs',   LA:'Los Angeles Rams',     LAC:'Los Angeles Chargers',
  LV:'Las Vegas Raiders',    MIA:'Miami Dolphins',      MIN:'Minnesota Vikings',
  NE:'New England Patriots', NO:'New Orleans Saints',   NYG:'New York Giants',
  NYJ:'New York Jets',       PHI:'Philadelphia Eagles', PIT:'Pittsburgh Steelers',
  SEA:'Seattle Seahawks',    SF:'San Francisco 49ers',  TB:'Tampa Bay Buccaneers',
  TEN:'Tennessee Titans',    WAS:'Washington Commanders',
}

const RAW_COLORS = {
  KC:'#E31837', SF:'#AA0000', BUF:'#00338D', DAL:'#003594', PHI:'#004C54',
  BAL:'#241773', CIN:'#FB4F14', DET:'#0076B6', MIA:'#008E97', GB:'#203731',
  SEA:'#69BE28', LAC:'#0080C6', NYJ:'#125740', NE:'#002244', PIT:'#FFB612',
  HOU:'#03202F', JAX:'#006778', IND:'#002C5F', TEN:'#4B92DB', LV:'#A5ACAF',
  DEN:'#FB4F14', LA:'#003594',  ARI:'#97233F', ATL:'#A71930', CAR:'#0085CA',
  NO:'#D3BC8D',  TB:'#D50A0A',  WAS:'#5A1414', CHI:'#0B162A', MIN:'#4F2683',
  NYG:'#0B2265', CLE:'#311D00',
}

const DISPLAY_OVERRIDES = {
  CHI:'#4a6fa5', NE:'#4a7abf',  NYG:'#3a5fa5', NYJ:'#2d7a5a',
  DAL:'#4a78cc', BUF:'#2255b8', BAL:'#6644bb', IND:'#3366cc',
  LA: '#4a78cc', GB: '#3a7a4a', HOU:'#4a7a8a', CLE:'#7a4a2a',
  WAS:'#aa3333', PHI:'#2a7a6a', SF: '#cc2222', MIN:'#7744bb',
  ATL:'#cc2233', ARI:'#cc2233',
}

function teamColor(team) {
  return DISPLAY_OVERRIDES[team] || RAW_COLORS[team] || '#4a80ff'
}

function Rule({ color = 'var(--gold)', style = {} }) {
  return (
    <div style={{
      height: 1,
      background: `linear-gradient(90deg, transparent, ${color} 20%, ${color} 80%, transparent)`,
      ...style,
    }} />
  )
}

function Super({ children, color = 'var(--gold)', style = {} }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)', fontSize: 10,
      letterSpacing: '0.2em', color, textTransform: 'uppercase',
      marginBottom: 4, ...style,
    }}>
      {children}
    </div>
  )
}

function Pill({ label, value, color = 'var(--silver)' }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '6px 14px', borderLeft: '1px solid var(--border)',
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em' }}>
        {label}
      </span>
      <span style={{ fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 16, color }}>
        {value}
      </span>
    </div>
  )
}

function ProbBar({ homeProb, homeTeam, awayTeam }) {
  const homePct = Math.round(homeProb * 100)
  const awayPct = 100 - homePct
  const hColor  = teamColor(homeTeam)
  const aColor  = teamColor(awayTeam)
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: 'flex', height: 3, background: 'var(--bg)' }}>
        <div style={{
          width: `${homePct}%`,
          background: `linear-gradient(90deg, ${hColor}cc, ${hColor})`,
          transition: 'width 0.5s ease',
        }} />
        <div style={{ flex: 1, background: `linear-gradient(90deg, ${aColor}, ${aColor}cc)` }} />
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        marginTop: 5, fontFamily: 'var(--font-mono)', fontSize: 10,
      }}>
        <span style={{ color: homePct > 50 ? 'var(--silver-lt)' : 'var(--text-dim)' }}>
          {homeTeam}  {homePct}%
        </span>
        <span style={{ color: awayPct > 50 ? 'var(--silver-lt)' : 'var(--text-dim)' }}>
          {awayPct}%  {awayTeam}
        </span>
      </div>
    </div>
  )
}

// ── Conviction helpers ────────────────────────────────────────────────────────
// Conviction = how strongly the model disagrees with Vegas
// Based purely on spread_gap size — larger gap = stronger disagreement
function getConviction(gap) {
  if (gap == null) return null
  const absGap = Math.abs(gap)
  const MIN_GAP = 3.0
  const MAX_GAP = 8.0
  const pct   = Math.round(Math.max(0, Math.min(100, ((absGap - MIN_GAP) / (MAX_GAP - MIN_GAP)) * 100)))
  const label = pct >= 65 ? 'HIGH' : pct >= 30 ? 'MEDIUM' : 'LOW'
  const color = pct >= 65 ? 'var(--green)' : pct >= 30 ? 'var(--gold)' : 'var(--text-dim)'
  const fill  = pct >= 65 ? '#00aa55'      : pct >= 30 ? '#c9a84c'     : '#5a6070'
  return { pct, label, color, fill }
}

// ── Game Card ─────────────────────────────────────────────────────────────────
function GameCard({ game, idx }) {
  const homeWin      = game.predicted_winner === game.home_team
  const gap          = game.spread_gap
  const hasEdge      = gap !== null && Math.abs(gap) >= 3.0
  const isAtsBet     = hasEdge && game.div_game !== 1
  const winner       = game.predicted_winner
  const loser        = homeWin ? game.away_team : game.home_team
  const isHistory    = !!game.is_historical
  const modelCorrect = isHistory && game.actual_winner
    ? game.predicted_winner === game.actual_winner
    : null


  const topBorder =
    modelCorrect === true  ? 'var(--green)'  :
    modelCorrect === false ? 'var(--red)'    :
    isAtsBet               ? 'var(--gold)'   :
    teamColor(game.home_team)

  return (
    <div className="fadein" style={{
      animationDelay: `${idx * 0.035}s`,
      background: 'var(--bg2)',
      border: '1px solid var(--border)',
      borderTop: `2px solid ${topBorder}`,
      marginBottom: 8,
      overflow: 'hidden',
    }}>

      {/* ── History result bar ── */}
      {isHistory && game.actual_winner && (
        <div style={{
          display: 'flex', justifyContent: 'flex-end', alignItems: 'center',
          gap: 12, padding: '8px 18px',
          borderBottom: '1px solid var(--border)',
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.15em',
            color: modelCorrect ? 'var(--green)' : 'var(--red)',
          }}>
            {modelCorrect ? '✓ CORRECT' : '✗ WRONG'}
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.1em' }}>
            ACTUAL: <span style={{ color: 'var(--silver-lt)' }}>{game.actual_winner}</span>
            {game.result != null && (
              <span style={{ marginLeft: 6 }}>
                ({game.result > 0 ? game.home_team : game.away_team} by {Math.abs(game.result)})
              </span>
            )}
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)' }}>
            {game.gameday}
          </span>
        </div>
      )}

      {/* ── Pick section ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20, padding: '14px 18px 12px' }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
            <span style={{
              fontFamily: 'var(--font-head)', fontWeight: 700,
              fontSize: 28, lineHeight: 1, color: 'var(--white)', letterSpacing: '0.02em',
            }}>
              {winner}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gold)', letterSpacing: '0.2em' }}>
              {homeWin ? 'HOME' : 'AWAY'}  ·  PICK
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 2 }}>
            <span style={{
              fontFamily: 'var(--font-head)', fontWeight: 400,
              fontSize: 18, lineHeight: 1, color: 'var(--text-dim)', letterSpacing: '0.02em',
            }}>
              {loser}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.1em' }}>
              {homeWin ? 'AWAY' : 'HOME'}
            </span>
          </div>
          <ProbBar homeProb={game.home_win_prob} homeTeam={game.home_team} awayTeam={game.away_team} />
        </div>

        <div style={{ width: 1, alignSelf: 'stretch', background: 'var(--border)', flexShrink: 0 }} />

        <div style={{ display: 'flex', alignItems: 'stretch' }}>
          <Pill
            label="MODEL"
            value={game.model_spread != null ? (game.model_spread > 0 ? `+${game.model_spread.toFixed(1)}` : game.model_spread.toFixed(1)) : '—'}
            color={game.model_spread > 0 ? 'var(--silver-lt)' : game.model_spread < 0 ? 'var(--silver)' : 'var(--text-dim)'}
          />
          <Pill
            label="VEGAS"
            value={game.vegas_spread != null ? (game.vegas_spread > 0 ? `+${game.vegas_spread.toFixed(1)}` : game.vegas_spread.toFixed(1)) : '—'}
            color={game.vegas_spread != null ? 'var(--silver)' : 'var(--text-dim)'}
          />
          {hasEdge && (
            <Pill label="EDGE" value={`${gap > 0 ? '+' : ''}${gap.toFixed(1)}`} color="var(--gold)" />
          )}
        </div>

        {!isHistory && (
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 9,
            color: 'var(--text-dim)', letterSpacing: '0.1em', textAlign: 'right',
            minWidth: 60,
          }}>
            <div>{game.gameday}</div>
            <div>{game.gametime}</div>
          </div>
        )}
      </div>


    </div>
  )
}

// ── Props Card ────────────────────────────────────────────────────────────────
function PropCard({ player, idx, statKey, statLabel, threshold = 25 }) {
  const [propLine, setPropLine] = useState('')

  const pred    = player[`pred_${statKey}`]
  const gap     = propLine !== '' && pred != null ? pred - parseFloat(propLine) : null
  const absGap  = gap !== null ? Math.abs(gap) : null
  const nobet   = absGap !== null && absGap < threshold
  const isOver  = gap !== null && gap >= threshold
  const isUnder = gap !== null && gap <= -threshold

  const recColor =
    nobet   ? 'var(--text-dim)' :
    isOver  ? 'var(--green)'    :
    isUnder ? 'var(--red)'      : 'var(--text-dim)'

  const recLabel =
    propLine === '' ? '—'      :
    nobet           ? 'NO BET' :
    isOver          ? '▲ OVER' : '▼ UNDER'

  const confidence =
    absGap === null          ? ''       :
    absGap >= threshold * 2  ? 'HIGH'   :
    absGap >= threshold      ? 'MEDIUM' : ''

  const accent = teamColor(player.team)

  return (
    <div className="fadein" style={{
      animationDelay: `${idx * 0.04}s`,
      background: 'var(--bg2)',
      border: '1px solid var(--border)',
      borderLeft: `3px solid ${accent}`,
      marginBottom: 8,
      padding: '14px 18px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 2 }}>
            <span style={{
              fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 22,
              color: 'var(--white)', letterSpacing: '0.02em',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {player.player_name}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: accent, letterSpacing: '0.15em', flexShrink: 0 }}>
              {player.team}
            </span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.12em' }}>
            VS {player.opponent}  ·  {player.is_home ? 'HOME' : 'AWAY'}  ·  WK {player.week}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 0, borderLeft: '1px solid var(--border)' }}>
          {player.pred_passing_yards   != null && <Pill label="PROJ YDS"  value={player.pred_passing_yards?.toFixed(0)   ?? '—'} color="var(--silver-lt)" />}
          {player.pred_passing_tds     != null && <Pill label="PROJ TDS"  value={player.pred_passing_tds?.toFixed(1)     ?? '—'} color="var(--silver)" />}
          {player.pred_attempts        != null && <Pill label="PROJ ATT"  value={player.pred_attempts?.toFixed(0)         ?? '—'} color="var(--silver)" />}
          {player.pred_rushing_yards   != null && <Pill label="RUSH YDS"  value={player.pred_rushing_yards?.toFixed(0)   ?? '—'} color="var(--silver-lt)" />}
          {player.pred_rushing_tds     != null && <Pill label="RUSH TDS"  value={player.pred_rushing_tds?.toFixed(1)     ?? '—'} color="var(--silver)" />}
          {player.pred_carries         != null && <Pill label="CARRIES"   value={player.pred_carries?.toFixed(0)          ?? '—'} color="var(--silver)" />}
          {player.pred_receiving_yards != null && <Pill label="REC YDS"   value={player.pred_receiving_yards?.toFixed(0) ?? '—'} color="var(--silver-lt)" />}
          {player.pred_receiving_tds   != null && <Pill label="REC TDS"   value={player.pred_receiving_tds?.toFixed(1)   ?? '—'} color="var(--silver)" />}
          {player.pred_receptions      != null && <Pill label="REC"       value={player.pred_receptions?.toFixed(1)       ?? '—'} color="var(--silver)" />}
          {player.pred_targets         != null && <Pill label="TARGETS"   value={player.pred_targets?.toFixed(1)          ?? '—'} color="var(--silver)" />}
          <Pill label="IMP TOTAL" value={player.implied_team_total?.toFixed(1) ?? '—'} color="var(--text-dim)" />
        </div>

        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          gap: 6, borderLeft: '1px solid var(--border)', paddingLeft: 16, minWidth: 120,
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em' }}>
            {statLabel} LINE
          </span>
          <input
            type="number" placeholder="—" value={propLine}
            onChange={e => setPropLine(e.target.value)}
            style={{
              fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 18,
              width: 80, textAlign: 'center',
              background: 'var(--bg3)', color: 'var(--white)',
              border: '1px solid var(--border2)', padding: '4px 8px', outline: 'none',
            }}
          />
          {propLine !== '' && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 14, letterSpacing: '0.12em', color: recColor }}>
                {recLabel}
              </div>
              {confidence && (
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: recColor, letterSpacing: '0.15em', opacity: 0.7 }}>
                  {confidence}  {gap !== null ? `(${gap > 0 ? '+' : ''}${gap.toFixed(1)})` : ''}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Props Tab ─────────────────────────────────────────────────────────────────
const POSITION_CONFIG = {
  qb: { label:'QB PROPS', super:'Quarterback projections', endpoint:'/props/qb', statKey:'passing_yards',  statLabel:'PASS YDS', threshold:25, legend:['▲ OVER ≥ +25 yds','▼ UNDER ≤ −25 yds','NO BET < 25 yds gap'] },
  rb: { label:'RB PROPS', super:'Running back projections', endpoint:'/props/rb', statKey:'rushing_yards',  statLabel:'RUSH YDS', threshold:15, legend:['▲ OVER ≥ +15 yds','▼ UNDER ≤ −15 yds','NO BET < 15 yds gap'] },
  wr: { label:'WR PROPS', super:'Wide receiver projections', endpoint:'/props/wr', statKey:'receiving_yards', statLabel:'REC YDS',  threshold:15, legend:['▲ OVER ≥ +15 yds','▼ UNDER ≤ −15 yds','NO BET < 15 yds gap'] },
}

function PropsTab() {
  const [pos, setPos]         = useState('qb')
  const [data, setData]       = useState({})
  const [loading, setLoading] = useState(false)
  const [status, setStatus]   = useState('')
  const cfg = POSITION_CONFIG[pos]

  useEffect(() => {
    if (data[pos]) return
    setLoading(true); setStatus('')
    axios.get(`${API}${cfg.endpoint}`)
      .then(r => { setData(d => ({ ...d, [pos]: r.data })); setLoading(false) })
      .catch(() => { setStatus('No prop data. Run weekly update first.'); setLoading(false) })
  }, [pos])

  const players = data[pos] || []
  const posTabStyle = (id) => ({
    fontFamily:'var(--font-head)', fontWeight:500, fontSize:12, letterSpacing:'0.12em', padding:'6px 18px',
    border:'none', cursor:'pointer', transition:'all 0.15s',
    background: pos === id ? 'var(--gold)' : 'transparent',
    color: pos === id ? 'var(--bg)' : 'var(--text-dim)',
  })

  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-end', marginBottom:18 }}>
        <div>
          <Super>{cfg.super}</Super>
          <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:32, color:'var(--white)', letterSpacing:'0.03em', lineHeight:1 }}>{cfg.label}</div>
          <div style={{ marginTop:6, fontFamily:'var(--font-mono)', fontSize:10, color:'var(--text-dim)', letterSpacing:'0.12em' }}>Enter the book's prop line to get an OVER / UNDER recommendation</div>
        </div>
        <div style={{ display:'flex', flexDirection:'column', alignItems:'flex-end', gap:8 }}>
          <div style={{ display:'flex', border:'1px solid var(--border2)', overflow:'hidden' }}>
            {['qb','rb','wr'].map(p => <button key={p} onClick={() => setPos(p)} style={posTabStyle(p)}>{p.toUpperCase()}</button>)}
          </div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.15em', textAlign:'right' }}>
            <div style={{ color:'var(--green)', marginBottom:2 }}>{cfg.legend[0]}</div>
            <div style={{ color:'var(--red)' }}>{cfg.legend[1]}</div>
            <div style={{ marginTop:2 }}>{cfg.legend[2]}</div>
          </div>
        </div>
      </div>
      <Rule style={{ marginBottom:18 }} />
      {status && <div style={{ fontFamily:'var(--font-mono)', fontSize:10, letterSpacing:'0.1em', color:'var(--text-dim)', marginBottom:14 }}>{status}</div>}
      {loading ? (
        <div style={{ textAlign:'center', marginTop:80, fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-dim)', letterSpacing:'0.2em' }}>LOADING...</div>
      ) : players.length === 0 ? (
        <div style={{ textAlign:'center', marginTop:80, color:'var(--text-dim)' }}>
          <div style={{ marginBottom:8, fontFamily:'var(--font-mono)', fontSize:11, letterSpacing:'0.15em' }}>NO PROP DATA AVAILABLE</div>
          <div style={{ fontSize:13 }}>Run the weekly update from the PICKS tab to generate projections.</div>
        </div>
      ) : players.map((p, i) => <PropCard key={`${p.player_name}-${p.team}-${i}`} player={p} idx={i} statKey={cfg.statKey} statLabel={cfg.statLabel} threshold={cfg.threshold} />)}
    </div>
  )
}

// ── Picks Tab ─────────────────────────────────────────────────────────────────
function WinnersTab() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading]         = useState(true)
  const [refreshing, setRefreshing]   = useState(false)
  const [status, setStatus]           = useState('')
  const [historical, setHistorical]   = useState(false)
  const [histSeason, setHistSeason]   = useState(2024)
  const [histWeek,   setHistWeek]     = useState(1)

  const seasons = Array.from({ length: 25 }, (_, i) => 2025 - i)
  const weeks   = Array.from({ length: 22 }, (_, i) => i + 1)

  const normalizeHistorical = (games) => games.map(g => {
    const home_win_prob = g.home_win_prob != null
      ? g.home_win_prob
      : (() => { const e = (g.model_spread || 0) * 25; return 1 / (1 + Math.pow(10, -e / 400)) })()
    return { ...g, predicted_winner: g.model_winner, home_win_prob, is_historical: true }
  })

  const loadLive = async () => {
    setLoading(true)
    try {
      const res = await axios.get(`${API}/predictions`)
      setPredictions(res.data)
    } catch { setStatus('Connection error. Verify api.py is running.') }
    finally  { setLoading(false) }
  }

  const loadHistoricalWeek = async (season, week) => {
    setLoading(true)
    try {
      const res = await axios.get(`${API}/predictions/historical`, { params: { season, week } })
      setPredictions(normalizeHistorical(res.data))
    } catch { setStatus('Connection error. Verify api.py is running.') }
    finally  { setLoading(false) }
  }

  useEffect(() => { loadLive() }, [])

  const refresh = async () => {
    setRefreshing(true); setStatus('Syncing data...')
    try {
      await axios.post(`${API}/refresh`)
      const poll = async () => {
        const s = await axios.get(`${API}/refresh/status`)
        if (s.data.running) { setTimeout(poll, 2000) }
        else {
          setStatus(s.data.last_result === 'success' ? 'Updated.' : s.data.last_result)
          setRefreshing(false); loadLive()
        }
      }
      poll()
    } catch (e) { setStatus(e.response?.data?.detail || 'Refresh failed'); setRefreshing(false) }
  }

  const atsBets = predictions.filter(p =>
    p.spread_gap != null && Math.abs(p.spread_gap) >= 3.0 && p.div_game !== 1
  )

  const selectStyle = {
    fontFamily:'var(--font-head)', fontWeight:500, fontSize:13, letterSpacing:'0.08em', padding:'5px 10px',
    background:'var(--bg3)', color:'var(--white)', border:'1px solid var(--border2)',
    cursor:'pointer', outline:'none', appearance:'none', WebkitAppearance:'none',
  }

  return (
    <div>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-end', marginBottom:18 }}>
        <div>
          <Super>{historical ? `${histSeason} · WEEK ${histWeek}` : "Week's predictions"}</Super>
          <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:32, color:'var(--white)', letterSpacing:'0.03em', lineHeight:1 }}>
            GAME PICKS
          </div>
          {atsBets.length > 0 && !historical && (
            <div style={{ marginTop:6, display:'flex', alignItems:'center', gap:8 }}>
              <span style={{ color: 'var(--gold)', fontSize: 12 }}>★</span>
              <span style={{ fontFamily:'var(--font-mono)', fontSize:10, color:'var(--gold)', letterSpacing:'0.15em' }}>
                {atsBets.length} ATS bet{atsBets.length !== 1 ? 's' : ''} this week
              </span>
            </div>
          )}
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <div style={{ display:'flex', border:'1px solid var(--border2)', overflow:'hidden' }}>
            {[{ id:false, label:'LIVE' },{ id:true, label:'HISTORY' }].map(m => (
              <button key={String(m.id)} onClick={() => { setHistorical(m.id); if (!m.id) { setStatus(''); loadLive() } }} style={{
                fontFamily:'var(--font-head)', fontWeight:500, fontSize:11, letterSpacing:'0.12em', padding:'6px 14px',
                border:'none', cursor:'pointer', transition:'all 0.15s',
                background: historical === m.id ? 'var(--gold)' : 'transparent',
                color: historical === m.id ? 'var(--bg)' : 'var(--text-dim)',
              }}>{m.label}</button>
            ))}
          </div>
          {historical ? (
            <>
              <select value={histSeason} onChange={e => setHistSeason(+e.target.value)} style={selectStyle}>
                {seasons.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <select value={histWeek} onChange={e => setHistWeek(+e.target.value)} style={selectStyle}>
                {weeks.map(w => <option key={w} value={w}>WK {w}</option>)}
              </select>
              <button onClick={() => { setStatus(''); loadHistoricalWeek(histSeason, histWeek) }} style={{
                fontFamily:'var(--font-head)', fontWeight:500, fontSize:13, letterSpacing:'0.15em', padding:'8px 20px',
                background:'transparent', border:'1px solid var(--gold)', color:'var(--gold)', cursor:'pointer',
              }}>GO</button>
            </>
          ) : (
            <button onClick={refresh} disabled={refreshing} style={{
              fontFamily:'var(--font-head)', fontWeight:500, fontSize:13, letterSpacing:'0.15em', padding:'8px 20px',
              background:'transparent',
              border:`1px solid ${refreshing ? 'var(--border2)' : 'var(--gold)'}`,
              color: refreshing ? 'var(--text-dim)' : 'var(--gold)',
              cursor: refreshing ? 'default' : 'pointer', transition:'all 0.15s',
            }}>{refreshing ? 'SYNCING...' : 'REFRESH'}</button>
          )}
        </div>
      </div>

      <Rule style={{ marginBottom:18 }} />

      {status && (
        <div style={{ fontFamily:'var(--font-mono)', fontSize:10, letterSpacing:'0.1em', color: status.includes('Updated') ? 'var(--green)' : 'var(--text-dim)', marginBottom:14 }}>
          {status}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign:'center', marginTop:80, fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-dim)', letterSpacing:'0.2em' }}>LOADING...</div>
      ) : predictions.length === 0 ? (
        <div style={{ textAlign:'center', marginTop:80, color:'var(--text-dim)' }}>No predictions. Click Refresh to run the weekly update.</div>
      ) : (
        predictions.map((g, i) => <GameCard key={g.game_id} game={g} idx={i} />)
      )}
    </div>
  )
}

// ── Rankings Tab ──────────────────────────────────────────────────────────────
function EloTab() {
  const [ratings, setRatings] = useState([])
  useEffect(() => { axios.get(`${API}/elo`).then(r => setRatings(r.data)).catch(() => {}) }, [])
  const max = ratings[0]?.elo ?? 1800
  const min = ratings[ratings.length - 1]?.elo ?? 1200

  return (
    <div>
      <Super>Current standings</Super>
      <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:32, color:'var(--white)', letterSpacing:'0.03em', lineHeight:1, marginBottom:18 }}>POWER RANKINGS</div>
      <Rule style={{ marginBottom:16 }} />
      <div style={{ display:'grid', gridTemplateColumns:'32px 220px 1fr 70px 48px', padding:'0 12px 8px', fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.15em', borderBottom:'1px solid var(--border)', marginBottom:4 }}>
        <span>RK</span><span>TEAM</span><span /><span style={{ textAlign:'right' }}>RATING</span><span style={{ textAlign:'right' }}>TIER</span>
      </div>
      {ratings.map((r, i) => {
        const pct  = ((r.elo - min) / (max - min)) * 100
        const tier = r.elo >= 1650 ? { label:'I',   color:'var(--gold)' }
                   : r.elo >= 1560 ? { label:'II',  color:'var(--silver-lt)' }
                   : r.elo >= 1470 ? { label:'III', color:'var(--silver)' }
                   : r.elo >= 1400 ? { label:'IV',  color:'var(--text-dim)' }
                   :                 { label:'V',   color:'#4a4a60' }
        const fullName = TEAM_NAMES[r.team] || r.team
        const accent   = teamColor(r.team)
        return (
          <div key={r.team} style={{ display:'grid', gridTemplateColumns:'32px 220px 1fr 70px 48px', alignItems:'center', padding:'7px 12px', borderBottom:'1px solid var(--border)', borderLeft:`2px solid ${i < 5 ? accent : 'transparent'}`, background: i < 5 ? 'rgba(201,168,76,0.025)' : 'transparent' }}>
            <span style={{ fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-dim)' }}>{i + 1}</span>
            <div>
              <div style={{ fontFamily:'var(--font-head)', fontWeight:600, fontSize:15, color:'var(--white)', letterSpacing:'0.02em', lineHeight:1.1 }}>{fullName}</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.1em', marginTop:2 }}>{r.team}</div>
            </div>
            <div style={{ paddingRight:16 }}>
              <div style={{ height:2, background:'var(--bg)' }}>
                <div style={{ width:`${pct}%`, height:'100%', background:accent, minWidth:2, opacity:0.6 }} />
              </div>
            </div>
            <span style={{ fontFamily:'var(--font-mono)', fontSize:13, color:'var(--text)', textAlign:'right' }}>{Math.round(r.elo)}</span>
            <span style={{ fontFamily:'var(--font-mono)', fontSize:10, color:tier.color, textAlign:'right', letterSpacing:'0.1em' }}>{tier.label}</span>
          </div>
        )
      })}
    </div>
  )
}

// ── Stats Tab ─────────────────────────────────────────────────────────────────
function AboutTab() {
  const [stats, setStats] = useState([])
  useEffect(() => { axios.get(`${API}/stats`).then(r => setStats(r.data)).catch(() => {}) }, [])
  const allTime = stats.find(s => s.season === 'All-time')
  const edge    = allTime ? (allTime.model_accuracy - allTime.vegas_accuracy) * 100 : null
  const summaryStats = allTime ? [
    { label:'GAMES ANALYZED', value:allTime.games.toLocaleString(),                         color:'var(--silver-lt)' },
    { label:'MODEL ACCURACY', value:`${(allTime.model_accuracy * 100).toFixed(1)}%`,        color:'var(--silver-lt)' },
    { label:'VEGAS ACCURACY', value:`${(allTime.vegas_accuracy * 100).toFixed(1)}%`,        color:'var(--silver)' },
    { label:'EDGE VS VEGAS',  value:`${edge > 0 ? '+' : ''}${edge?.toFixed(1)}%`,           color: edge > 0 ? 'var(--gold)' : 'var(--red)' },
  ] : []

  return (
    <div style={{ maxWidth:760 }}>
      <Super>About this model</Super>
      <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:32, color:'var(--white)', letterSpacing:'0.03em', lineHeight:1, marginBottom:18 }}>JAGINATOR</div>
      <Rule style={{ marginBottom:20 }} />
      <p style={{ fontFamily:'var(--font-body)', fontWeight:300, fontSize:15, lineHeight:1.8, color:'var(--text)', marginBottom:28, borderLeft:'2px solid var(--gold-dk)', paddingLeft:16 }}>
        NFL game prediction engine trained on 26 seasons of data (1999–2025). Combines ELO ratings, player EPA metrics, team rolling stats, and injury data using a pure XGBoost architecture trained on ATS margin with walk-forward validation. Vegas spread is included as a peer feature — the model learns its own relationship with the line rather than anchoring to it. During January and February a separate playoff logistic regression model replaces the main model for playoff games.
      </p>
      {summaryStats.length > 0 && (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4, 1fr)', border:'1px solid var(--border)', marginBottom:28 }}>
          {summaryStats.map((s, i) => (
            <div key={s.label} style={{ padding:'16px', borderRight: i < 3 ? '1px solid var(--border)' : 'none' }}>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.15em', marginBottom:6 }}>{s.label}</div>
              <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:26, color:s.color, lineHeight:1 }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}
      <Super style={{ marginBottom:10 }}>Season-by-season record</Super>
      <div style={{ border:'1px solid var(--border)' }}>
        <div style={{ display:'grid', gridTemplateColumns:'70px 60px 80px 80px 70px', padding:'7px 14px', background:'var(--bg3)', borderBottom:'1px solid var(--border2)', fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.15em' }}>
          <span>SEASON</span><span style={{ textAlign:'right' }}>GAMES</span><span style={{ textAlign:'right' }}>MODEL</span><span style={{ textAlign:'right' }}>VEGAS</span><span style={{ textAlign:'right' }}>EDGE</span>
        </div>
        {stats.filter(s => s.season !== 'All-time').map((s) => {
          const rowEdge = (s.model_accuracy - s.vegas_accuracy) * 100
          return (
            <div key={s.season} style={{ display:'grid', gridTemplateColumns:'70px 60px 80px 80px 70px', padding:'5px 14px', borderBottom:'1px solid var(--border)', fontFamily:'var(--font-mono)', fontSize:12, background: rowEdge > 0 ? 'rgba(201,168,76,0.03)' : 'transparent' }}>
              <span style={{ color:'var(--silver)' }}>{s.season}</span>
              <span style={{ textAlign:'right', color:'var(--text-dim)' }}>{s.games}</span>
              <span style={{ textAlign:'right', color:'var(--text)' }}>{(s.model_accuracy * 100).toFixed(1)}%</span>
              <span style={{ textAlign:'right', color:'var(--text-dim)' }}>{(s.vegas_accuracy * 100).toFixed(1)}%</span>
              <span style={{ textAlign:'right', fontWeight: rowEdge > 0 ? 700 : 400, color: rowEdge > 0 ? 'var(--gold)' : 'var(--text-dim)' }}>{rowEdge > 0 ? '+' : ''}{rowEdge.toFixed(1)}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Strategy Tab helpers ──────────────────────────────────────────────────────
function StatBox({ label, value, color = 'var(--gold)' }) {
  return (
    <div style={{ padding:'14px 18px', background:'var(--bg3)', border:'1px solid var(--border)', borderTop:`2px solid ${color}`, flex:1 }}>
      <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.15em', marginBottom:6 }}>{label}</div>
      <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:22, color, lineHeight:1 }}>{value}</div>
    </div>
  )
}

function Section({ title, super: superLabel, children, accent = 'var(--gold)' }) {
  return (
    <div style={{ marginBottom:40 }}>
      <Super color={accent} style={{ marginBottom:6 }}>{superLabel}</Super>
      <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:24, color:'var(--white)', letterSpacing:'0.03em', lineHeight:1, marginBottom:14 }}>{title}</div>
      <Rule color={accent} style={{ marginBottom:18 }} />
      {children}
    </div>
  )
}

function Callout({ children, color = 'var(--gold-dk)' }) {
  return (
    <div style={{ borderLeft:`2px solid ${color}`, paddingLeft:16, marginBottom:16, fontFamily:'var(--font-body)', fontWeight:300, fontSize:14, lineHeight:1.8, color:'var(--text)' }}>
      {children}
    </div>
  )
}

function BetRow({ label, value, good = null }) {
  const color = good === true ? 'var(--green)' : good === false ? 'var(--red)' : 'var(--silver)'
  return (
    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'8px 14px', borderBottom:'1px solid var(--border)', fontFamily:'var(--font-mono)', fontSize:11 }}>
      <span style={{ color:'var(--text)', letterSpacing:'0.1em' }}>{label}</span>
      <span style={{ color, fontWeight: good !== null ? 700 : 400 }}>{value}</span>
    </div>
  )
}

// ── Strategy Tab ──────────────────────────────────────────────────────────────
function StrategyTab() {
  return (
    <div style={{ maxWidth:780 }}>
      <Super style={{ marginBottom:6 }}>How to use this tool</Super>
      <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:32, color:'var(--white)', letterSpacing:'0.03em', lineHeight:1, marginBottom:6 }}>BETTING STRATEGY</div>
      <div style={{ fontFamily:'var(--font-mono)', fontSize:10, color:'var(--text-dim)', letterSpacing:'0.12em', marginBottom:20 }}>Understanding the model, reading the signals, and knowing when to bet</div>
      <Rule style={{ marginBottom:32 }} />

      <Section title="THE MATH" super="Before anything else" accent="var(--gold)">
        <Callout>At -110 odds, you need to win 52.4% of bets just to break even after the vig. Everything else in this guide is secondary to understanding that number.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="BREAK-EVEN WIN RATE" value="52.4%" color="var(--gold)" />
          <StatBox label="VIG ON $100 BET"      value="$9.09"  color="var(--silver)" />
          <StatBox label="ROI AT 55% WIN RATE"  value="+4.5%" color="var(--green)" />
          <StatBox label="ROI AT 50% WIN RATE"  value="-4.5%" color="var(--red)" />
        </div>
        <div style={{ border:'1px solid var(--border)' }}>
          <BetRow label="Win" value="+$100 net" good={true} />
          <BetRow label="Lose" value="-$110 net" good={false} />
          <BetRow label="Break-even win rate" value="52.4%" />
          <BetRow label="55% win rate over 100 bets" value="+$450" good={true} />
          <BetRow label="50% win rate over 100 bets" value="-$550" good={false} />
        </div>
      </Section>

      <Section title="PLAYOFF MODEL" super="January–February" accent="var(--green)">
        <Callout>During playoff weeks the app automatically switches to a logistic regression model trained specifically on playoff games. The game cards look identical — the playoff model silently replaces the main model's winner prediction and win probability.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="PLAYOFF ACCURACY" value="68.9%"  color="var(--green)" />
          <StatBox label="VS VEGAS"          value="+5.5%"  color="var(--silver-lt)" />
          <StatBox label="CONF CHAMPIONSHIP" value="77.3%"  color="var(--gold)" />
          <StatBox label="SUPER BOWL"        value="68.2%"  color="var(--gold)" />
        </div>
        <div style={{ border:'1px solid var(--border)', marginBottom:16 }}>
          <BetRow label="Wild Card"           value="62.0% — moderate signal" />
          <BetRow label="Divisional"          value="72.7% — strong signal"    good={true} />
          <BetRow label="Conference Championship" value="77.3% — strongest signal" good={true} />
          <BetRow label="Super Bowl"          value="68.2% vs 50% Vegas baseline" good={true} />
        </div>
      </Section>

      <Section title="READING A GAME CARD" super="Spread model" accent="var(--blue-lt)">
        <Callout>Every card shows a pick — who the model thinks wins. The model and Vegas almost always agree on the winner. Where they differ is the margin: the model often projects a more extreme spread than Vegas. When that gap is large enough, it flags potential value.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="MODEL ACCURACY"  value="66.2%"    color="var(--silver-lt)" />
          <StatBox label="ATS WIN RATE"    value="54.8%"    color="var(--gold)" />
          <StatBox label="BETS / SEASON"   value="~23"      color="var(--silver)" />
          <StatBox label="EDGE THRESHOLD"  value="±3.0 pts" color="var(--gold)" />
        </div>
        <div style={{ border:'1px solid var(--border)', marginBottom:16 }}>
          <BetRow label="PICK"   value="Who the model thinks wins — always shown" />
          <BetRow label="MODEL"  value="Margin the model projects (e.g. PHI -4.9 = model thinks PHI wins by 4.9)" />
          <BetRow label="VEGAS"  value="The book's spread (e.g. PHI -1.5 = Vegas thinks PHI wins by 1.5)" />
          <BetRow label="EDGE"   value="Model minus Vegas — how much the model thinks the picked team is underpriced" />
        </div>
        <Callout color="var(--gold-dk)">This model is a starting point, not a final answer. Use it alongside your own knowledge — if the model sees a 4 point edge on a team but you know their star WR is out or they just played Thursday, trust your read. The model doesn't watch film. You do. The best bets combine a model edge with a reason you believe it.</Callout>
        <div style={{ border:'1px solid var(--border)', marginBottom:16, padding:'12px 14px', display:'flex', flexDirection:'column', gap:12 }}>
          <div>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--green)', letterSpacing:'0.12em', marginBottom:4 }}>EXAMPLE 1 — FAVORITE UNDERVALUED</div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--text)', lineHeight:1.6 }}>
              PHI vs KC. Vegas has PHI as a 1.5 point favorite. The model projects PHI wins by 4.9 — an EDGE of 3.4. Vegas thinks this is a close game. The model thinks PHI blows them out.
            </div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--silver-lt)', lineHeight:1.6, marginTop:6 }}>
              <strong>Trust the model:</strong> Bet PHI -1.5. You need PHI to win by 2 or more to cash. The model says they win by nearly 5 — plenty of cushion.
            </div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--silver)', lineHeight:1.6, marginTop:4 }}>
              <strong>Trust Vegas:</strong> No bet — if you think it's a 1.5 point game, the spread is already right.
            </div>
          </div>
          <div style={{ borderTop:'1px solid var(--border)', paddingTop:12 }}>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--green)', letterSpacing:'0.12em', marginBottom:4 }}>EXAMPLE 2 — UNDERDOG UNDERVALUED</div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--text)', lineHeight:1.6 }}>
              MIA vs CLE. Vegas has MIA favored by 3. The model only sees MIA winning by 0.7 — an EDGE of 3.2 on CLE. Vegas is giving CLE way more credit than the model thinks they deserve, but the model still likes MIA to win.
            </div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--silver-lt)', lineHeight:1.6, marginTop:6 }}>
              <strong>Trust the model:</strong> Bet CLE +3. The model has this as nearly a pick-em — CLE getting 3 free points is value. CLE wins or loses by 1 or 2 and you still cash.
            </div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--silver)', lineHeight:1.6, marginTop:4 }}>
              <strong>Trust Vegas:</strong> No bet — if you think MIA genuinely wins by 3+, the line is fair.
            </div>
          </div>
          <div style={{ borderTop:'1px solid var(--border)', paddingTop:12 }}>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.12em', marginBottom:4 }}>NO EDGE — CONSIDER PARLAYS</div>
            <div style={{ fontFamily:'var(--font-body)', fontSize:13, color:'var(--text-dim)', lineHeight:1.6 }}>
              No EDGE means the model and Vegas agree on the margin — no spread value identified. These games are still worth watching for moneyline parlays. The model's 66% accuracy on picking the winner means stacking 2–3 high-confidence picks into a parlay can produce value even without a spread edge. Stick to games where win probability is 65%+.
            </div>
          </div>
        </div>
      </Section>

      <Section title="QB PROPS" super="Passing yards model" accent="var(--silver)">
        <Callout>The model averages about 59 yards of error on QB passing yards. Use as a second opinion against the book's line, not standalone.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="MAE" value="59 yds" color="var(--silver)" />
          <StatBox label="BET THRESHOLD" value="±25 yds" color="var(--gold)" />
        </div>
        <div style={{ border:'1px solid var(--border)', marginBottom:16 }}>
          <BetRow label="Model 290, book 255" value="▲ OVER (+35)"  good={true} />
          <BetRow label="Model 220, book 255" value="▼ UNDER (-35)" good={true} />
          <BetRow label="Model 265, book 255" value="NO BET (+10)"  good={false} />
        </div>
      </Section>

      <Section title="RB PROPS" super="Rushing yards model" accent="var(--silver)">
        <Callout>Carries and receptions are the most reliable predictions. Rushing yards average about 27 yards of error.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="RUSH YDS MAE"  value="27 yds" color="var(--silver)" />
          <StatBox label="CARRIES MAE"   value="4.4"    color="var(--green)" />
          <StatBox label="BET THRESHOLD" value="±15 yds" color="var(--gold)" />
        </div>
        <div style={{ border:'1px solid var(--border)', marginBottom:16 }}>
          <BetRow label="Carries and receptions" value="Most reliable"        good={true} />
          <BetRow label="Rushing yards"          value="Use ±15 yd threshold" good={true} />
        </div>
      </Section>

      <Section title="WR PROPS" super="Receiving yards model" accent="var(--silver)">
        <Callout>Target share is stable week to week. The model captures scheme tendencies well.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="REC YDS MAE"    value="27 yds" color="var(--silver)" />
          <StatBox label="RECEPTIONS MAE" value="1.66"   color="var(--green)" />
          <StatBox label="BET THRESHOLD"  value="±15 yds" color="var(--gold)" />
        </div>
        <div style={{ border:'1px solid var(--border)', marginBottom:16 }}>
          <BetRow label="Receptions and targets" value="Most reliable"               good={true} />
          <BetRow label="Receiving yards"        value="Similar accuracy to rushing"  good={true} />
        </div>
      </Section>

      <Section title="BANKROLL" super="Money management" accent="var(--gold)">
        <Callout>Flat betting 1–3% per bet gives enough runway to let edge play out over hundreds of bets. Scale up on HIGH confidence picks, down on MEDIUM.</Callout>
        <div style={{ display:'flex', gap:1, marginBottom:16 }}>
          <StatBox label="RECOMMENDED UNIT" value="1–2%"   color="var(--gold)" />
          <StatBox label="HIGH CONF UNIT"    value="2–3%"   color="var(--green)" />
          <StatBox label="NEVER EXCEED"      value="5%"     color="var(--red)" />
          <StatBox label="TRACK EVERY BET"   value="Always" color="var(--silver-lt)" />
        </div>
        <div style={{ border:'1px solid var(--border)', marginBottom:16 }}>
          <BetRow label="$1,000 bankroll at 1% unit"       value="$10 per bet" />
          <BetRow label="55% win rate, 100 bets"            value="+$450"          good={true} />
          <BetRow label="10-game losing streak at 1% units" value="-$100 drawdown" good={true} />
          <BetRow label="10-game losing streak at 5% units" value="-$500 drawdown" good={false} />
        </div>
      </Section>
    </div>
  )
}

// ── App shell ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("winners")
  const tabs = [
    { id:"winners",  label:"PICKS"    },
    { id:"props",    label:"PROPS"    },
    { id:"elo",      label:"RANKINGS" },
    { id:"about",    label:"STATS"    },
    { id:"strategy", label:"STRATEGY" },
  ]

  return (
    <>
      <style>{STYLES}</style>
      <div style={{ minHeight:'100vh', background:'var(--bg)' }}>
        <div style={{ height:2, background:'linear-gradient(90deg, var(--gold-dk), var(--gold) 30%, var(--gold-lt) 50%, var(--gold) 70%, var(--gold-dk))' }} />
        <div style={{ background:'var(--bg2)', borderBottom:'1px solid var(--border2)' }}>
          <div style={{ maxWidth:1000, margin:'0 auto', padding:'0 28px', display:'flex', alignItems:'stretch' }}>
            <div style={{ fontFamily:'var(--font-head)', fontWeight:700, fontSize:22, letterSpacing:'0.12em', color:'var(--gold)', padding:'16px 28px 16px 0', marginRight:4, borderRight:'1px solid var(--border)', display:'flex', alignItems:'center', userSelect:'none' }}>
              JAGINATOR
            </div>
            <div style={{ display:'flex' }}>
              {tabs.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={{
                  fontFamily:'var(--font-head)', fontWeight:500, fontSize:13, letterSpacing:'0.15em', padding:'0 20px',
                  border:'none', background:'none',
                  color: tab === t.id ? 'var(--white)' : 'var(--text-dim)',
                  borderBottom: tab === t.id ? '2px solid var(--gold)' : '2px solid transparent',
                  cursor:'pointer', transition:'color 0.15s',
                }}>{t.label}</button>
              ))}
            </div>
            <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-dim)', letterSpacing:'0.2em' }}>
              NFL  ·  1999–2025
            </div>
          </div>
        </div>
        <div style={{ maxWidth:1000, margin:'0 auto', padding:'32px 28px 64px' }}>
          {tab === 'winners'  && <WinnersTab />}
          {tab === 'props'    && <PropsTab />}
          {tab === 'elo'      && <EloTab />}
          {tab === 'about'    && <AboutTab />}
          {tab === 'strategy' && <StrategyTab />}
        </div>
      </div>
    </>
  )
}