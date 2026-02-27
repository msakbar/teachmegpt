export function TopBar({ activeIndex, onMenuOpen }) {
  const inHero = activeIndex === -1;

  return (
    <div
      className={`md:hidden sticky top-0 z-50 bg-[#faf9f6]/92 backdrop-blur-[12px] border-b border-[rgba(0,0,0,0.06)] px-3 py-2 flex items-center gap-1.5 transition-opacity duration-300${inHero ? " opacity-0 pointer-events-none" : ""}`}
    >
      <button
        onClick={onMenuOpen}
        aria-label="Open navigation"
        className="bg-[rgba(0,0,0,0.03)] border-none text-ink-secondary p-2.5 rounded-lg cursor-pointer text-sm leading-none min-w-[44px] min-h-[44px] flex items-center justify-center"
      >
        &#9776;
      </button>
    </div>
  );
}
