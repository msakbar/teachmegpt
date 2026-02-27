import { useState, useCallback, useRef } from "react";
import { SECTIONS } from "./data/sections";
import { useIsMobile } from "./hooks/useIsMobile";
import { useActiveSection } from "./hooks/useActiveSection";
import { Sidebar } from "./components/layout/Sidebar";
import { TopBar } from "./components/layout/TopBar";
import { MobileDrawer } from "./components/layout/MobileDrawer";
import { SectionContainer } from "./components/layout/SectionContainer";
import { Hero } from "./components/layout/Hero";

export default function App() {
  const isMobile = useIsMobile();
  const [drawerOpen, setDrawerOpen] = useState(false);
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

      {/* Main content */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
        <TopBar
          activeIndex={activeIndex}
          totalSections={SECTIONS.length}
          onPrev={() => handleNav(Math.max(0, activeIndex - 1))}
          onNext={() =>
            handleNav(Math.min(SECTIONS.length - 1, Math.max(0, activeIndex) + 1))
          }
          onMenuOpen={() => setDrawerOpen(true)}
          isMobile={isMobile}
        />

        <Hero onStartReading={() => handleNav(0)} />
        <SectionContainer setSectionRef={setSectionRef} />
      </div>
    </div>
  );
}
