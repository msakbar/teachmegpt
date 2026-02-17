import { motion, AnimatePresence } from "framer-motion";

export function MobileDrawer({
  isOpen,
  onClose,
  sections,
  activeIndex,
  onNavigate,
}) {
  return (
    <AnimatePresence>
      {isOpen ? (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/30 z-[200]"
            aria-hidden="true"
          />

          {/* Drawer */}
          <motion.nav
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            aria-label="Sections"
            className="fixed top-0 left-0 bottom-0 w-[min(300px,82vw)] bg-page z-[201] overflow-y-auto border-r border-[rgba(0,0,0,0.06)]"
          >
            {/* Header */}
            <div className="px-4 pt-5 pb-4 border-b border-[rgba(0,0,0,0.06)] flex justify-between items-start">
              <div>
                <div className="font-serif text-[15px] font-semibold text-ink mb-0.5">
                  microgpt.py
                </div>
                <div className="font-serif text-[12px] text-ink-secondary">
                  by Karpathy
                </div>
              </div>
              <button
                onClick={onClose}
                aria-label="Close navigation"
                className="bg-[rgba(0,0,0,0.03)] border-none text-ink-tertiary text-sm w-9 h-9 rounded-lg cursor-pointer flex items-center justify-center"
              >
                &#10005;
              </button>
            </div>

            {/* Items */}
            <div className="py-3 px-2">
              {sections.map((section, i) => {
                const isActive = i === activeIndex;
                return (
                  <button
                    key={section.id}
                    onClick={() => {
                      onNavigate(i);
                      onClose();
                    }}
                    className={`w-full text-left px-3 py-3 rounded-lg border-none cursor-pointer mb-0.5 min-h-[44px] font-serif text-[14px] leading-snug ${
                      isActive
                        ? "font-bold text-ink border-l-[3px] border-accent-fill bg-transparent"
                        : "font-normal text-ink-secondary bg-transparent"
                    }`}
                  >
                    {section.title}
                  </button>
                );
              })}
            </div>
          </motion.nav>
        </>
      ) : null}
    </AnimatePresence>
  );
}
