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
  const hColor = teamColor(homeTeam)
  const aColor = teamColor(awayTeam)
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

function GameCard({ game, idx }) {
  const homeWin      = game.predicted_winner === game.home_team
  const gap          = game.spread_gap
  const hasEdge      = gap !== null && Math.abs(gap) >= 2.5
  const winner       = game.predicted_winner
  const loser        = homeWin ? game.away_team : game.home_team
  const isHistory    = !!game.is_historical
  const modelCorrect = isHistory && game.actual_winner
    ? game.predicted_winner === game.actual_winner
    : null

  const topBorder =
    modelCorrect === true  ? 'var(--green)' :
    modelCorrect === false ? 'var(--red)'   :
    hasEdge ? 'var(--gold)' : teamColor(game.home_team)

  return (
    <div className="fadein" style={{
      animationDelay: `${idx * 0.035}s`,
      background: 'var(--bg2)',
      border: '1px solid var(--border)',
      borderTop: `2px solid ${topBorder}`,
      marginBottom: 8,
      padding: '14px 18px 12px',
    }}>
      {isHistory && game.actual_winner && (
        <div style={{
          display: 'flex', justifyContent: 'flex-end', alignItems: 'center',
          gap: 12, marginBottom: 10,
          paddingBottom: 8, borderBottom: '1px solid var(--border)',
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

      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
            <span style={{
              fontFamily: 'var(--font-head)', fontWeight: 700,
              fontSize: 28, lineHeight: 1, color: 'var(--white)', letterSpacing: '0.02em',
            }}>
              {winner}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--gold)', letterSpacing: '0.2em' }}>
              {homeWin ? 'HOME' : 'AWAY'}  ·  FAVORED
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

// ── Generic Props Card ───────────────────────────────────────────────────────
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
        {/* Player info */}
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

        {/* Stat pills — position specific */}
        <div style={{ display: 'flex', gap: 0, borderLeft: '1px solid var(--border)' }}>
          {player.pred_passing_yards  != null && <Pill label="PROJ YDS"  value={player.pred_passing_yards?.toFixed(0)  ?? '—'} color="var(--silver-lt)" />}
          {player.pred_passing_tds    != null && <Pill label="PROJ TDS"  value={player.pred_passing_tds?.toFixed(1)    ?? '—'} color="var(--silver)" />}
          {player.pred_attempts       != null && <Pill label="PROJ ATT"  value={player.pred_attempts?.toFixed(0)        ?? '—'} color="var(--silver)" />}
          {player.pred_rushing_yards  != null && <Pill label="RUSH YDS"  value={player.pred_rushing_yards?.toFixed(0)  ?? '—'} color="var(--silver-lt)" />}
          {player.pred_rushing_tds    != null && <Pill label="RUSH TDS"  value={player.pred_rushing_tds?.toFixed(1)    ?? '—'} color="var(--silver)" />}
          {player.pred_carries        != null && <Pill label="CARRIES"   value={player.pred_carries?.toFixed(0)         ?? '—'} color="var(--silver)" />}
          {player.pred_receiving_yards!= null && <Pill label="REC YDS"   value={player.pred_receiving_yards?.toFixed(0)?? '—'} color="var(--silver-lt)" />}
          {player.pred_receiving_tds  != null && <Pill label="REC TDS"   value={player.pred_receiving_tds?.toFixed(1)  ?? '—'} color="var(--silver)" />}
          {player.pred_receptions     != null && <Pill label="REC"       value={player.pred_receptions?.toFixed(1)      ?? '—'} color="var(--silver)" />}
          {player.pred_targets        != null && <Pill label="TARGETS"   value={player.pred_targets?.toFixed(1)         ?? '—'} color="var(--silver)" />}
          <Pill label="IMP TOTAL" value={player.implied_team_total?.toFixed(1) ?? '—'} color="var(--text-dim)" />
        </div>

        {/* Prop line input */}
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          gap: 6, borderLeft: '1px solid var(--border)', paddingLeft: 16, minWidth: 120,
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em' }}>
            {statLabel} LINE
          </span>
          <input
            type="number"
            placeholder="—"
            value={propLine}
            onChange={e => setPropLine(e.target.value)}
            style={{
              fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 18,
              width: 80, textAlign: 'center',
              background: 'var(--bg3)', color: 'var(--white)',
              border: '1px solid var(--border2)', padding: '4px 8px',
              outline: 'none',
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
  qb: {
    label: 'QB PROPS', super: 'Quarterback projections',
    endpoint: '/props/qb',
    statKey: 'passing_yards', statLabel: 'PASS YDS', threshold: 25,
    legend: ['▲ OVER ≥ +25 yds', '▼ UNDER ≤ −25 yds', 'NO BET < 25 yds gap'],
  },
  rb: {
    label: 'RB PROPS', super: 'Running back projections',
    endpoint: '/props/rb',
    statKey: 'rushing_yards', statLabel: 'RUSH YDS', threshold: 15,
    legend: ['▲ OVER ≥ +15 yds', '▼ UNDER ≤ −15 yds', 'NO BET < 15 yds gap'],
  },
  wr: {
    label: 'WR PROPS', super: 'Wide receiver projections',
    endpoint: '/props/wr',
    statKey: 'receiving_yards', statLabel: 'REC YDS', threshold: 15,
    legend: ['▲ OVER ≥ +15 yds', '▼ UNDER ≤ −15 yds', 'NO BET < 15 yds gap'],
  },
}

function PropsTab() {
  const [pos, setPos]         = useState('qb')
  const [data, setData]       = useState({})
  const [loading, setLoading] = useState(false)
  const [status, setStatus]   = useState('')

  const cfg = POSITION_CONFIG[pos]

  useEffect(() => {
    if (data[pos]) return
    setLoading(true)
    setStatus('')
    axios.get(`${API}${cfg.endpoint}`)
      .then(r => { setData(d => ({ ...d, [pos]: r.data })); setLoading(false) })
      .catch(() => { setStatus('No prop data — run weekly update first.'); setLoading(false) })
  }, [pos])

  const players = data[pos] || []

  const posTabStyle = (id) => ({
    fontFamily: 'var(--font-head)', fontWeight: 500,
    fontSize: 12, letterSpacing: '0.12em', padding: '6px 18px',
    border: 'none', cursor: 'pointer', transition: 'all 0.15s',
    background: pos === id ? 'var(--gold)' : 'transparent',
    color: pos === id ? 'var(--bg)' : 'var(--text-dim)',
  })

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 18 }}>
        <div>
          <Super>{cfg.super}</Super>
          <div style={{
            fontFamily: 'var(--font-head)', fontWeight: 700,
            fontSize: 32, color: 'var(--white)', letterSpacing: '0.03em', lineHeight: 1,
          }}>
            {cfg.label}
          </div>
          <div style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.12em' }}>
            Enter the book's prop line to get an OVER / UNDER recommendation
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8 }}>
          {/* Position sub-tabs */}
          <div style={{ display: 'flex', border: '1px solid var(--border2)', overflow: 'hidden' }}>
            {['qb','rb','wr'].map(p => (
              <button key={p} onClick={() => setPos(p)} style={posTabStyle(p)}>
                {p.toUpperCase()}
              </button>
            ))}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em', textAlign: 'right' }}>
            <div style={{ color: 'var(--green)', marginBottom: 2 }}>{cfg.legend[0]}</div>
            <div style={{ color: 'var(--red)' }}>{cfg.legend[1]}</div>
            <div style={{ marginTop: 2 }}>{cfg.legend[2]}</div>
          </div>
        </div>
      </div>

      <Rule style={{ marginBottom: 18 }} />

      {status && (
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em', color: 'var(--text-dim)', marginBottom: 14 }}>
          {status}
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: 'center', marginTop: 80, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)', letterSpacing: '0.2em' }}>
          LOADING...
        </div>
      ) : players.length === 0 ? (
        <div style={{ textAlign: 'center', marginTop: 80, color: 'var(--text-dim)' }}>
          <div style={{ marginBottom: 8, fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.15em' }}>
            NO PROP DATA AVAILABLE
          </div>
          <div style={{ fontSize: 13 }}>
            Run the weekly update from the PICKS tab to generate projections.
          </div>
        </div>
      ) : (
        players.map((p, i) => (
          <PropCard
            key={`${p.player_name}-${p.team}-${i}`}
            player={p} idx={i}
            statKey={cfg.statKey}
            statLabel={cfg.statLabel}
            threshold={cfg.threshold}
          />
        ))
      )}
    </div>
  )
}

// ── Existing tabs (unchanged) ─────────────────────────────────────────────────// ── Existing tabs (unchanged) ─────────────────────────────────────────────────
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
    const spreadToProb = (spread) => {
      const eloDiff = spread * 25
      return 1 / (1 + Math.pow(10, -eloDiff / 400))
    }
    const home_win_prob = spreadToProb(g.model_spread)
    return { ...g, predicted_winner: g.model_winner, home_win_prob, is_historical: true }
  })

  const loadLive = async () => {
    setLoading(true)
    try {
      const res = await axios.get(`${API}/predictions`)
      setPredictions(res.data)
    } catch {
      setStatus('Connection error — verify api.py is running')
    } finally {
      setLoading(false)
    }
  }

  const loadHistoricalWeek = async (season, week) => {
    setLoading(true)
    try {
      const res = await axios.get(`${API}/predictions/historical`, { params: { season, week } })
      setPredictions(normalizeHistorical(res.data))
    } catch {
      setStatus('Connection error — verify api.py is running')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadLive() }, [])

  const refresh = async () => {
    setRefreshing(true)
    setStatus('Syncing data...')
    try {
      await axios.post(`${API}/refresh`)
      const poll = async () => {
        const s = await axios.get(`${API}/refresh/status`)
        if (s.data.running) {
          setTimeout(poll, 2000)
        } else {
          setStatus(s.data.last_result === 'success' ? 'Updated.' : s.data.last_result)
          setRefreshing(false)
          loadLive()
        }
      }
      poll()
    } catch (e) {
      setStatus(e.response?.data?.detail || 'Refresh failed')
      setRefreshing(false)
    }
  }

  const edgeGames = predictions.filter(p => p.spread_gap != null && Math.abs(p.spread_gap) >= 2.5)

  const selectStyle = {
    fontFamily: 'var(--font-head)', fontWeight: 500,
    fontSize: 13, letterSpacing: '0.08em', padding: '5px 10px',
    background: 'var(--bg3)', color: 'var(--white)',
    border: '1px solid var(--border2)', cursor: 'pointer',
    outline: 'none', appearance: 'none', WebkitAppearance: 'none',
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 18 }}>
        <div>
          <Super>{historical ? `${histSeason} · WEEK ${histWeek}` : "Week's predictions"}</Super>
          <div style={{
            fontFamily: 'var(--font-head)', fontWeight: 700,
            fontSize: 32, color: 'var(--white)', letterSpacing: '0.03em', lineHeight: 1,
          }}>
            GAME PICKS
          </div>
          {edgeGames.length > 0 && (
            <div style={{
              marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 10,
              color: 'var(--gold)', letterSpacing: '0.15em',
            }}>
              {edgeGames.length} value edge{edgeGames.length !== 1 ? 's' : ''} detected
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ display: 'flex', border: '1px solid var(--border2)', overflow: 'hidden' }}>
            {[{ id: false, label: 'LIVE' }, { id: true, label: 'HISTORY' }].map(m => (
              <button key={String(m.id)} onClick={() => {
                setHistorical(m.id)
                if (!m.id) { setStatus(''); loadLive() }
              }} style={{
                fontFamily: 'var(--font-head)', fontWeight: 500,
                fontSize: 11, letterSpacing: '0.12em', padding: '6px 14px',
                border: 'none', cursor: 'pointer', transition: 'all 0.15s',
                background: historical === m.id ? 'var(--gold)' : 'transparent',
                color: historical === m.id ? 'var(--bg)' : 'var(--text-dim)',
              }}>
                {m.label}
              </button>
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
                fontFamily: 'var(--font-head)', fontWeight: 500,
                fontSize: 13, letterSpacing: '0.15em', padding: '8px 20px',
                background: 'transparent', border: '1px solid var(--gold)',
                color: 'var(--gold)', cursor: 'pointer',
              }}>GO</button>
            </>
          ) : (
            <button onClick={refresh} disabled={refreshing} style={{
              fontFamily: 'var(--font-head)', fontWeight: 500,
              fontSize: 13, letterSpacing: '0.15em', padding: '8px 20px',
              background: 'transparent',
              border: `1px solid ${refreshing ? 'var(--border2)' : 'var(--gold)'}`,
              color: refreshing ? 'var(--text-dim)' : 'var(--gold)',
              cursor: refreshing ? 'default' : 'pointer', transition: 'all 0.15s',
            }}>
              {refreshing ? 'SYNCING...' : 'REFRESH'}
            </button>
          )}
        </div>
      </div>

      <Rule style={{ marginBottom: 18 }} />

      {status && (
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.1em',
          color: status.includes('Updated') ? 'var(--green)' : 'var(--text-dim)',
          marginBottom: 14,
        }}>
          {status}
        </div>
      )}

      {loading ? (
        <div style={{
          textAlign: 'center', marginTop: 80,
          fontFamily: 'var(--font-mono)', fontSize: 11,
          color: 'var(--text-dim)', letterSpacing: '0.2em',
        }}>LOADING...</div>
      ) : predictions.length === 0 ? (
        <div style={{ textAlign: 'center', marginTop: 80, color: 'var(--text-dim)' }}>
          No predictions — click Refresh to run the weekly update.
        </div>
      ) : (
        predictions.map((g, i) => <GameCard key={g.game_id} game={g} idx={i} />)
      )}
    </div>
  )
}

function EloTab() {
  const [ratings, setRatings] = useState([])

  useEffect(() => {
    axios.get(`${API}/elo`).then(r => setRatings(r.data)).catch(() => {})
  }, [])

  const max = ratings[0]?.elo ?? 1800
  const min = ratings[ratings.length - 1]?.elo ?? 1200

  return (
    <div>
      <Super>Current standings</Super>
      <div style={{
        fontFamily: 'var(--font-head)', fontWeight: 700,
        fontSize: 32, color: 'var(--white)', letterSpacing: '0.03em',
        lineHeight: 1, marginBottom: 18,
      }}>
        POWER RANKINGS
      </div>
      <Rule style={{ marginBottom: 16 }} />

      <div style={{
        display: 'grid', gridTemplateColumns: '32px 220px 1fr 70px 48px',
        padding: '0 12px 8px',
        fontFamily: 'var(--font-mono)', fontSize: 9,
        color: 'var(--text-dim)', letterSpacing: '0.15em',
        borderBottom: '1px solid var(--border)', marginBottom: 4,
      }}>
        <span>RK</span><span>TEAM</span><span /><span style={{ textAlign: 'right' }}>RATING</span><span style={{ textAlign: 'right' }}>TIER</span>
      </div>

      {ratings.map((r, i) => {
        const pct = ((r.elo - min) / (max - min)) * 100
        const tier =
          r.elo >= 1650 ? { label: 'I',   color: 'var(--gold)' }      :
          r.elo >= 1560 ? { label: 'II',  color: 'var(--silver-lt)' } :
          r.elo >= 1470 ? { label: 'III', color: 'var(--silver)' }    :
          r.elo >= 1400 ? { label: 'IV',  color: 'var(--text-dim)' }  :
                          { label: 'V',   color: '#4a4a60' }
        const fullName = TEAM_NAMES[r.team] || r.team
        const accent = teamColor(r.team)

        return (
          <div key={r.team} style={{
            display: 'grid', gridTemplateColumns: '32px 220px 1fr 70px 48px',
            alignItems: 'center', padding: '7px 12px',
            borderBottom: '1px solid var(--border)',
            borderLeft: `2px solid ${i < 5 ? accent : 'transparent'}`,
            background: i < 5 ? 'rgba(201,168,76,0.025)' : 'transparent',
          }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>{i + 1}</span>
            <div>
              <div style={{ fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 15, color: 'var(--white)', letterSpacing: '0.02em', lineHeight: 1.1 }}>
                {fullName}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.1em', marginTop: 2 }}>
                {r.team}
              </div>
            </div>
            <div style={{ paddingRight: 16 }}>
              <div style={{ height: 2, background: 'var(--bg)' }}>
                <div style={{ width: `${pct}%`, height: '100%', background: accent, minWidth: 2, opacity: 0.6 }} />
              </div>
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text)', textAlign: 'right' }}>{Math.round(r.elo)}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: tier.color, textAlign: 'right', letterSpacing: '0.1em' }}>{tier.label}</span>
          </div>
        )
      })}
    </div>
  )
}

function AboutTab() {
  const [stats, setStats] = useState([])
  useEffect(() => { axios.get(`${API}/stats`).then(r => setStats(r.data)).catch(() => {}) }, [])

  const allTime = stats.find(s => s.season === 'All-time')
  const edge = allTime ? (allTime.model_accuracy - allTime.vegas_accuracy) * 100 : null

  const summaryStats = allTime ? [
    { label: 'GAMES ANALYZED', value: allTime.games.toLocaleString(), color: 'var(--silver-lt)' },
    { label: 'MODEL ACCURACY', value: `${(allTime.model_accuracy * 100).toFixed(1)}%`, color: 'var(--silver-lt)' },
    { label: 'VEGAS ACCURACY', value: `${(allTime.vegas_accuracy * 100).toFixed(1)}%`, color: 'var(--silver)' },
    { label: 'EDGE VS VEGAS',  value: `${edge > 0 ? '+' : ''}${edge?.toFixed(1)}%`, color: edge > 0 ? 'var(--gold)' : 'var(--red)' },
  ] : []

  return (
    <div style={{ maxWidth: 760 }}>
      <Super>About this model</Super>
      <div style={{
        fontFamily: 'var(--font-head)', fontWeight: 700,
        fontSize: 32, color: 'var(--white)', letterSpacing: '0.03em', lineHeight: 1, marginBottom: 18,
      }}>
        JAGINATOR
      </div>
      <Rule style={{ marginBottom: 20 }} />

      <p style={{
        fontFamily: 'var(--font-body)', fontWeight: 300, fontSize: 15, lineHeight: 1.8,
        color: 'var(--text)', marginBottom: 28,
        borderLeft: '2px solid var(--gold-dk)', paddingLeft: 16,
      }}>
        NFL game prediction engine trained on 26 seasons of data (1999–2025).
        Combines ELO ratings, player EPA metrics, team rolling stats, and weather conditions
        using a stacked XGBoost + Ridge regression architecture with walk-forward validation.
      </p>

      {summaryStats.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', border: '1px solid var(--border)', marginBottom: 28 }}>
          {summaryStats.map((s, i) => (
            <div key={s.label} style={{ padding: '16px', borderRight: i < 3 ? '1px solid var(--border)' : 'none' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em', marginBottom: 6 }}>
                {s.label}
              </div>
              <div style={{ fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 26, color: s.color, lineHeight: 1 }}>
                {s.value}
              </div>
            </div>
          ))}
        </div>
      )}

      <Super style={{ marginBottom: 10 }}>Season-by-season record</Super>
      <div style={{ border: '1px solid var(--border)' }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '70px 60px 80px 80px 70px',
          padding: '7px 14px', background: 'var(--bg3)',
          borderBottom: '1px solid var(--border2)',
          fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em',
        }}>
          <span>SEASON</span><span style={{ textAlign: 'right' }}>GAMES</span>
          <span style={{ textAlign: 'right' }}>MODEL</span><span style={{ textAlign: 'right' }}>VEGAS</span>
          <span style={{ textAlign: 'right' }}>EDGE</span>
        </div>
        {stats.filter(s => s.season !== 'All-time').map((s) => {
          const rowEdge = (s.model_accuracy - s.vegas_accuracy) * 100
          return (
            <div key={s.season} style={{
              display: 'grid', gridTemplateColumns: '70px 60px 80px 80px 70px',
              padding: '5px 14px', borderBottom: '1px solid var(--border)',
              fontFamily: 'var(--font-mono)', fontSize: 12,
              background: rowEdge > 0 ? 'rgba(201,168,76,0.03)' : 'transparent',
            }}>
              <span style={{ color: 'var(--silver)' }}>{s.season}</span>
              <span style={{ textAlign: 'right', color: 'var(--text-dim)' }}>{s.games}</span>
              <span style={{ textAlign: 'right', color: 'var(--text)' }}>{(s.model_accuracy * 100).toFixed(1)}%</span>
              <span style={{ textAlign: 'right', color: 'var(--text-dim)' }}>{(s.vegas_accuracy * 100).toFixed(1)}%</span>
              <span style={{
                textAlign: 'right', fontWeight: rowEdge > 0 ? 700 : 400,
                color: rowEdge > 0 ? 'var(--gold)' : 'var(--text-dim)',
              }}>
                {rowEdge > 0 ? '+' : ''}{rowEdge.toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}


// ── Strategy Tab ──────────────────────────────────────────────────────────────
function StatBox({ label, value, color = 'var(--gold)' }) {
  return (
    <div style={{
      padding: '14px 18px', background: 'var(--bg3)',
      border: '1px solid var(--border)', borderTop: `2px solid ${color}`,
      flex: 1,
    }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.15em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 22, color, lineHeight: 1 }}>
        {value}
      </div>
    </div>
  )
}

function Section({ title, super: superLabel, children, accent = 'var(--gold)' }) {
  return (
    <div style={{ marginBottom: 40 }}>
      <Super color={accent} style={{ marginBottom: 6 }}>{superLabel}</Super>
      <div style={{
        fontFamily: 'var(--font-head)', fontWeight: 700,
        fontSize: 24, color: 'var(--white)', letterSpacing: '0.03em',
        lineHeight: 1, marginBottom: 14,
      }}>
        {title}
      </div>
      <Rule color={accent} style={{ marginBottom: 18 }} />
      {children}
    </div>
  )
}

function Callout({ children, color = 'var(--gold-dk)' }) {
  return (
    <div style={{
      borderLeft: `2px solid ${color}`, paddingLeft: 16,
      marginBottom: 16,
      fontFamily: 'var(--font-body)', fontWeight: 300,
      fontSize: 14, lineHeight: 1.8, color: 'var(--text)',
    }}>
      {children}
    </div>
  )
}

function BetRow({ label, value, good = null }) {
  const color = good === true ? 'var(--green)' : good === false ? 'var(--red)' : 'var(--silver)'
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 14px', borderBottom: '1px solid var(--border)',
      fontFamily: 'var(--font-mono)', fontSize: 11,
    }}>
      <span style={{ color: 'var(--text-dim)', letterSpacing: '0.1em' }}>{label}</span>
      <span style={{ color, fontWeight: good !== null ? 700 : 400 }}>{value}</span>
    </div>
  )
}

function StrategyTab() {
  return (
    <div style={{ maxWidth: 780 }}>
      <Super style={{ marginBottom: 6 }}>How to use this tool</Super>
      <div style={{
        fontFamily: 'var(--font-head)', fontWeight: 700,
        fontSize: 32, color: 'var(--white)', letterSpacing: '0.03em', lineHeight: 1, marginBottom: 6,
      }}>
        BETTING STRATEGY
      </div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.12em', marginBottom: 20 }}>
        Understanding the model, reading the signals, and knowing when to bet
      </div>
      <Rule style={{ marginBottom: 32 }} />

      {/* ── The Math ── */}
      <Section title="THE MATH" super="Before anything else" accent="var(--gold)">
        <Callout>
          At -110 odds (standard US), you must win 52.4% of bets just to break even after the vig.
          This is the single most important number in sports betting. Everything else is secondary.
        </Callout>
        <div style={{ display: 'flex', gap: 1, marginBottom: 16 }}>
          <StatBox label="BREAK-EVEN WIN RATE" value="52.4%" color="var(--gold)" />
          <StatBox label="VIG ON $100 BET" value="$9.09" color="var(--silver)" />
          <StatBox label="ROI AT 55% WIN RATE" value="+4.5%" color="var(--green)" />
          <StatBox label="ROI AT 50% WIN RATE" value="-4.5%" color="var(--red)" />
        </div>
        <Callout>
          Implied probability at -110: 52.4%. If you believe a team has a 55%+ chance of covering,
          the bet has positive expected value. Below 52.4%, you are paying the book to gamble.
        </Callout>
        <div style={{ border: '1px solid var(--border)', marginTop: 8 }}>
          <BetRow label="BET $110 TO WIN $100 AT -110" value="" />
          <BetRow label="Win $100 net" value="+$100" good={true} />
          <BetRow label="Lose $110 net" value="-$110" good={false} />
          <BetRow label="Break-even win rate" value="52.4%" />
          <BetRow label="At 55% win rate on 100 bets" value="+$450 profit" good={true} />
          <BetRow label="At 50% win rate on 100 bets" value="-$550 loss" good={false} />
        </div>
      </Section>

      {/* ── Game Picks ── */}
      <Section title="GAME PICKS" super="Spread model" accent="var(--blue-lt)">
        <Callout>
          The model predicts the home team margin using ELO ratings, player EPA, rolling team stats,
          and injury data across 26 seasons. It's then stacked with Vegas's own spread using a Ridge
          meta-model, which is how it stays competitive.
        </Callout>
        <div style={{ display: 'flex', gap: 1, marginBottom: 16 }}>
          <StatBox label="MODEL ACCURACY" value="66.6%" color="var(--silver-lt)" />
          <StatBox label="EDGE VS VEGAS" value="+0.1%" color="var(--gold)" />
          <StatBox label="EDGE THRESHOLD" value="±2.5 pts" color="var(--silver)" />
          <StatBox label="SHARP FILTER" value="100% ATS*" color="var(--green)" />
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-dim)', letterSpacing: '0.1em', marginBottom: 16 }}>
          *Sharp filter: 3 bets, 2022 sample. Not statistically significant on its own.
        </div>

        <div style={{ fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 14, color: 'var(--silver-lt)', letterSpacing: '0.08em', marginBottom: 10 }}>
          HOW TO READ A GAME PICK
        </div>
        <div style={{ border: '1px solid var(--border)', marginBottom: 16 }}>
          <BetRow label="MODEL spread" value="Home team projected margin" />
          <BetRow label="VEGAS spread" value="Book's line (negative = home favored)" />
          <BetRow label="EDGE" value="Model minus Vegas — the disagreement" />
          <BetRow label="Gold border card" value="Edge ≥ 2.5 pts detected" />
        </div>

        <div style={{ fontFamily: 'var(--font-head)', fontWeight: 600, fontSize: 14, color: 'var(--silver-lt)', letterSpacing: '0.08em', marginBottom: 10 }}>
          WHEN TO BET
        </div>
        <div style={{ border: '1px solid var(--border)', marginBottom: 16 }}>
          <BetRow label="Edge ≥ 2.5 pts AND model + Vegas agree on winner" value="Potential bet" good={true} />
          <BetRow label="Edge ≥ 2.5 pts AND line has moved toward model" value="Stronger signal" good={true} />
          <BetRow label="Edge < 2.5 pts" value="Skip — noise, not signal" good={false} />
          <BetRow label="Weeks 17-18" value="Skip — rest + motivation variance" good={false} />
          <BetRow label="Division games" value="Caution — familiarity reduces edge" />
        </div>

        <Callout color="var(--blue-lt)">
          The model is trained on public data. Vegas has sharper information. Treat the edge as a
          starting point, not a guarantee. The value is in finding games where Vegas may have
          mispriced — not in blindly fading Vegas every time.
        </Callout>
      </Section>

      {/* ── QB Props ── */}
      <Section title="QB PROPS" super="Passing yards model" accent="var(--silver)">
        <Callout>
          QB passing yards have an all-time MAE of 59 yards — meaning the model is off by about
          59 yards on average. Vegas is sharper here (~40-45 yards MAE). The model is most useful
          as a sanity check against the book's line, not as a standalone predictor.
        </Callout>
        <div style={{ display: 'flex', gap: 1, marginBottom: 16 }}>
          <StatBox label="MODEL MAE" value="59 yds" color="var(--silver)" />
          <StatBox label="WITHIN 25 YDS" value="26.9%" color="var(--silver)" />
          <StatBox label="BET THRESHOLD" value="±25 yds" color="var(--gold)" />
          <StatBox label="HIGH CONFIDENCE" value="±50 yds" color="var(--gold-lt)" />
        </div>

        <div style={{ border: '1px solid var(--border)', marginBottom: 16 }}>
          <BetRow label="Model predicts 290, book says 255" value="▲ OVER (+35)" good={true} />
          <BetRow label="Model predicts 220, book says 255" value="▼ UNDER (-35)" good={true} />
          <BetRow label="Model predicts 265, book says 255" value="NO BET (+10)" good={false} />
        </div>

        <Callout>
          Key features: implied team total, rolling passing yards, air yards per attempt, CPOE, and
          opponent pass defense rank. Implied team total is the strongest single feature — a QB on
          a team implied for 28 points will throw more than one implied for 17.
        </Callout>
      </Section>

      {/* ── RB Props ── */}
      <Section title="RB PROPS" super="Rushing yards model" accent="var(--silver)">
        <Callout>
          RB rushing yards MAE of 27 yards — better than QB props relative to the stat range.
          Carries (4.4 MAE) and receptions (1.4 MAE) are essentially perfect. The bell cow back
          is identified automatically by most carries over the last 4 games.
        </Callout>
        <div style={{ display: 'flex', gap: 1, marginBottom: 16 }}>
          <StatBox label="RUSH YDS MAE" value="27 yds" color="var(--silver)" />
          <StatBox label="CARRIES MAE" value="4.4" color="var(--green)" />
          <StatBox label="RUSH TDS (±1)" value="92.7%" color="var(--green)" />
          <StatBox label="BET THRESHOLD" value="±15 yds" color="var(--gold)" />
        </div>

        <div style={{ border: '1px solid var(--border)', marginBottom: 16 }}>
          <BetRow label="Carries and receptions" value="Highest confidence — near perfect" good={true} />
          <BetRow label="Rushing yards" value="Good — use ±15 yd threshold" good={true} />
          <BetRow label="Rushing TDs" value="Useful — 92.7% within 1 TD" good={true} />
          <BetRow label="Receiving yards" value="Solid for pass-catching backs" />
        </div>

        <Callout>
          The threshold is tighter than QB (15 yards vs 25) because RB rushing yards vary less
          game to game. A predicted gap of 15+ yards is more meaningful here than the same gap
          on a QB passing yards prop.
        </Callout>
      </Section>

      {/* ── WR Props ── */}
      <Section title="WR PROPS" super="Receiving yards model" accent="var(--silver)">
        <Callout>
          WR props are the most reliable bets in the model. Target share, air yards share, and WOPR
          are highly stable week to week — the model captures scheme tendencies better than
          game-to-game variance. Receptions and targets are essentially perfect.
        </Callout>
        <div style={{ display: 'flex', gap: 1, marginBottom: 16 }}>
          <StatBox label="REC YDS MAE" value="27 yds" color="var(--silver)" />
          <StatBox label="RECEPTIONS MAE" value="1.66" color="var(--green)" />
          <StatBox label="REC TDS (±1)" value="95.8%" color="var(--green)" />
          <StatBox label="BET THRESHOLD" value="±15 yds" color="var(--gold)" />
        </div>

        <div style={{ border: '1px solid var(--border)', marginBottom: 16 }}>
          <BetRow label="Receptions and targets" value="Best bets — near perfect accuracy" good={true} />
          <BetRow label="Receiving TDs" value="74.7% exact — very strong" good={true} />
          <BetRow label="Receiving yards" value="Good — same accuracy as RB rushing" good={true} />
          <BetRow label="Top 3 WRs shown per team" value="Sorted by target share" />
        </div>

        <Callout>
          Target share is the single most predictive WR feature. A WR getting 28% of team targets
          consistently will continue to get roughly that — the model leans on this stability.
          WR receptions and targets are the highest-confidence prop bets in the entire app.
        </Callout>
      </Section>

      {/* ── Bankroll ── */}
      <Section title="BANKROLL" super="Money management" accent="var(--gold)">
        <Callout>
          The single most common mistake is betting too much per game. Even a profitable system
          will go on losing streaks. Flat betting 1-3% of your bankroll per bet survives variance
          long enough to let the edge compound.
        </Callout>
        <div style={{ display: 'flex', gap: 1, marginBottom: 16 }}>
          <StatBox label="RECOMMENDED UNIT" value="1-2%" color="var(--gold)" />
          <StatBox label="MAX PER GAME" value="3%" color="var(--silver)" />
          <StatBox label="NEVER BET MORE THAN" value="5%" color="var(--red)" />
          <StatBox label="TRACK EVERY BET" value="Always" color="var(--silver-lt)" />
        </div>

        <div style={{ border: '1px solid var(--border)', marginBottom: 16 }}>
          <BetRow label="$1,000 bankroll at 1% unit" value="$10 per bet" />
          <BetRow label="55% win rate, 100 bets" value="+$450 profit" good={true} />
          <BetRow label="10-game losing streak chance at 55%" value="~0.3% — it will happen" />
          <BetRow label="10-game losing streak at 5% units" value="-$500 drawdown" good={false} />
          <BetRow label="10-game losing streak at 1% units" value="-$100 drawdown" good={true} />
        </div>

        <Callout color="var(--gold-dk)">
          This tool gives you an edge, not a guarantee. No model beats the book every week.
          The goal is to be right more often than 52.4% over hundreds of bets — not to win every
          single one. Patience and discipline matter more than any single prediction.
        </Callout>
      </Section>
    </div>
  )
}

// ── App shell ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("winners")
  const tabs = [
    { id: "winners",  label: "PICKS"    },
    { id: "props",    label: "PROPS"    },
    { id: "elo",      label: "RANKINGS" },
    { id: "about",    label: "STATS"    },
    { id: "strategy", label: "STRATEGY" },
  ]

  return (
    <>
      <style>{STYLES}</style>
      <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
        <div style={{
          height: 2,
          background: 'linear-gradient(90deg, var(--gold-dk), var(--gold) 30%, var(--gold-lt) 50%, var(--gold) 70%, var(--gold-dk))',
        }} />

        <div style={{ background: 'var(--bg2)', borderBottom: '1px solid var(--border2)' }}>
          <div style={{ maxWidth: 1000, margin: '0 auto', padding: '0 28px', display: 'flex', alignItems: 'stretch' }}>
            <div style={{
              fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 22,
              letterSpacing: '0.12em', color: 'var(--gold)',
              padding: '16px 28px 16px 0', marginRight: 4,
              borderRight: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', userSelect: 'none',
            }}>
              JAGINATOR
            </div>
            <div style={{ display: 'flex' }}>
              {tabs.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={{
                  fontFamily: 'var(--font-head)', fontWeight: 500,
                  fontSize: 13, letterSpacing: '0.15em', padding: '0 20px',
                  border: 'none', background: 'none',
                  color: tab === t.id ? 'var(--white)' : 'var(--text-dim)',
                  borderBottom: tab === t.id ? '2px solid var(--gold)' : '2px solid transparent',
                  cursor: 'pointer', transition: 'color 0.15s',
                }}>
                  {t.label}
                </button>
              ))}
            </div>
            <div style={{
              marginLeft: 'auto', display: 'flex', alignItems: 'center',
              fontFamily: 'var(--font-mono)', fontSize: 9,
              color: 'var(--text-dim)', letterSpacing: '0.2em',
            }}>
              NFL  ·  1999–2025
            </div>
          </div>
        </div>

        <div style={{ maxWidth: 1000, margin: '0 auto', padding: '32px 28px 64px' }}>
          {tab === 'winners' && <WinnersTab />}
          {tab === 'props'   && <PropsTab />}
          {tab === 'elo'     && <EloTab />}
          {tab === 'about'    && <AboutTab />}
          {tab === 'strategy' && <StrategyTab />}
        </div>
      </div>
    </>
  )
}