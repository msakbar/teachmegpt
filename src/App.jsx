import { useState, useCallback, useRef, useEffect } from "react";
import { SECTIONS } from "./data/sections";
import { useIsMobile } from "./hooks/useIsMobile";
import { useActiveSection } from "./hooks/useActiveSection";
import { Sidebar } from "./components/layout/Sidebar";
import { MobileDrawer } from "./components/layout/MobileDrawer";
import { SectionContainer } from "./components/layout/SectionContainer";
import { Hero } from "./components/layout/Hero";

export default function App() {
  const isMobile = useIsMobile();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const scrollContainerRef = useRef(null);

  const { activeIndex, scrollToSection, setSectionRef } = useActiveSection(
    SECTIONS.length,
    scrollContainerRef
  );

  const handleNav = useCallback(
    (index) => {
      scrollToSection(index);
    },
    [scrollToSection]
  );

  // Scroll-based reading progress (mobile only)
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;

    let rafId;
    const onScroll = () => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const { scrollTop, scrollHeight, clientHeight } = el;
        const max = scrollHeight - clientHeight;
        setScrollProgress(max > 0 ? (scrollTop / max) * 100 : 0);
      });
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <div className="flex h-screen bg-page">
      {/* Desktop Sidebar */}
      {!isMobile ? (
        <Sidebar
          sections={SECTIONS}
          activeIndex={activeIndex}
          onNavigate={handleNav}
        />
      ) : null}

      {/* Mobile Drawer */}
      {isMobile ? (
        <MobileDrawer
          isOpen={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          sections={SECTIONS}
          activeIndex={activeIndex}
          onNavigate={handleNav}
        />
      ) : null}

      {/* Reading progress bar — mobile only */}
      <div
        className={`md:hidden fixed top-0 left-0 z-[100] h-[2px] bg-gradient-to-r from-accent-fill to-accent-bold transition-opacity duration-300 ${
          scrollProgress < 1 ? "opacity-0" : "opacity-100"
        }`}
        style={{ width: `${scrollProgress}%` }}
      />

      {/* Fixed ☰ menu button — mobile only, appears when a section is active */}
      <button
        onClick={() => setDrawerOpen(true)}
        aria-label="Open navigation"
        className={`md:hidden fixed top-3 right-5 z-[99] flex items-center justify-center w-9 h-9 rounded-md bg-[rgba(0,0,0,0.04)] text-[16px] text-ink-secondary leading-none cursor-pointer border-none active:opacity-70 transition-opacity duration-300 ${
          activeIndex < 0 ? "opacity-0 pointer-events-none" : "opacity-100"
        }`}
      >
        &#9776;
      </button>

      {/* Main content */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
        <Hero onStartReading={() => handleNav(0)} />
        <SectionContainer
          setSectionRef={setSectionRef}
        />
      </div>
    </div>
  );
}
