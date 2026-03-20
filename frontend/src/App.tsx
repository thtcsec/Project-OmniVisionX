import { NavLink, Route, Routes } from "react-router-dom";
import { HomePage } from "@/pages/HomePage";

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-omni-border bg-omni-surface/80 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
          <NavLink to="/" className="font-semibold text-zinc-100 tracking-tight">
            Project OmniVision
          </NavLink>
          <nav className="flex gap-4 text-sm text-omni-muted">
            <NavLink
              to="/"
              className={({ isActive }) =>
                isActive ? "text-omni-accent" : "hover:text-zinc-300"
              }
              end
            >
              Home
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
