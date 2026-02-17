import { useState, useCallback } from "react";
import { SECTIONS } from "./data/sections";
import { useIsMobile } from "./hooks/useIsMobile";
import { useActiveSection } from "./hooks/useActiveSection";
import { Sidebar } from "./components/layout/Sidebar";
import { TopBar } from "./components/layout/TopBar";
import { MobileDrawer } from "./components/layout/MobileDrawer";
import { SectionContainer } from "./components/layout/SectionContainer";

export default function App() {
  const isMobile = useIsMobile();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const { activeIndex, scrollToSection, setSectionRef } = useActiveSection(
    SECTIONS.length
  );

  const handleNav = useCallback(
    (index) => {
      scrollToSection(index);
    },
    [scrollToSection]
  );

  return (
    <div className="flex min-h-screen bg-page">
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
      <div className="flex-1 overflow-y-auto">
        <TopBar
          activeIndex={activeIndex}
          totalSections={SECTIONS.length}
          onPrev={() => handleNav(Math.max(0, activeIndex - 1))}
          onNext={() =>
            handleNav(Math.min(SECTIONS.length - 1, activeIndex + 1))
          }
          onMenuOpen={() => setDrawerOpen(true)}
          isMobile={isMobile}
        />

        <SectionContainer setSectionRef={setSectionRef} />
      </div>
    </div>
  );
}
