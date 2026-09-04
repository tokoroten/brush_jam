import { useEffect, useState, type JSX } from 'react';
import { Room } from './Room.js';

const NAME_KEY = 'brushjam.name';

function roomIdFromPath(pathname: string): string | null {
  const match = /^\/r\/([a-z0-9]{4,16})$/.exec(pathname);
  return match ? match[1]! : null;
}

export function App(): JSX.Element {
  const [path, setPath] = useState(location.pathname);
  const [name, setName] = useState(() => localStorage.getItem(NAME_KEY) ?? '');
  const [draftName, setDraftName] = useState(name);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    const onPop = (): void => setPath(location.pathname);
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const roomId = roomIdFromPath(path);

  const saveName = (value: string): void => {
    const trimmed = value.trim().slice(0, 24);
    if (!trimmed) return;
    localStorage.setItem(NAME_KEY, trimmed);
    setName(trimmed);
  };

  if (!name) {
    return (
      <div className="home">
        <h1>Brush Jam</h1>
        <p>Pick a name others will see.</p>
        <input value={draftName} placeholder="your name" onChange={(e) => setDraftName(e.target.value)} />
        <button onClick={() => saveName(draftName)}>Continue</button>
      </div>
    );
  }

  if (roomId) return <Room roomId={roomId} name={name} />;

  const createRoom = async (): Promise<void> => {
    setCreating(true);
    const res = await fetch('/api/rooms', { method: 'POST' });
    const { roomId: created } = (await res.json()) as { roomId: string };
    history.pushState(null, '', `/r/${created}`);
    setPath(`/r/${created}`);
    setCreating(false);
  };

  return (
    <div className="home">
      <h1>Brush Jam</h1>
      <p>
        Draw together on one big canvas while the AI keeps reinterpreting what you make. Create a room and send the URL to a
        couple of friends.
      </p>
      <button disabled={creating} onClick={() => void createRoom()}>
        {creating ? 'Creating...' : 'Create room'}
      </button>
      <p className="hint">
        Signed in as <strong>{name}</strong>{' '}
        <button
          onClick={() => {
            localStorage.removeItem(NAME_KEY);
            setName('');
          }}
        >
          change
        </button>
      </p>
    </div>
  );
}
