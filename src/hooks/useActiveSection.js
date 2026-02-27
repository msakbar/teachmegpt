import { useState, useEffect, useRef, useCallback } from "react";

export function useActiveSection(sectionCount, scrollContainerRef) {
  const [activeIndex, setActiveIndex] = useState(-1);
  const sectionRefs = useRef([]);
  const refCallbacks = useRef(new Map());
  const intersectingRef = useRef(new Set());
  const isScrollingRef = useRef(false);
  const scrollTimeoutRef = useRef(null);

  useEffect(() => {
    sectionRefs.current = sectionRefs.current.slice(0, sectionCount);
  }, [sectionCount]);

  // Track active section via IntersectionObserver
  useEffect(() => {
    const container = scrollContainerRef?.current;
    if (!container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (isScrollingRef.current) return;

        for (const entry of entries) {
          const idx = sectionRefs.current.indexOf(entry.target);
          if (idx === -1) continue;
          if (entry.isIntersecting) {
            intersectingRef.current.add(idx);
          } else {
            intersectingRef.current.delete(idx);
          }
        }

        if (intersectingRef.current.size === 0) {
          // No section visible — user is in the Hero
          setActiveIndex(-1);
        } else {
          // Pick the topmost (smallest index) intersecting section
          setActiveIndex(Math.min(...intersectingRef.current));
        }
      },
      { root: container, threshold: 0.15, rootMargin: "-60px 0px -40% 0px" }
    );

    const currentRefs = sectionRefs.current;
    currentRefs.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => {
      intersectingRef.current.clear();
      currentRefs.forEach((ref) => {
        if (ref) observer.unobserve(ref);
      });
    };
  }, [sectionCount, scrollContainerRef]);

  const scrollToSection = useCallback((index) => {
    const el = sectionRefs.current[index];
    const container = scrollContainerRef?.current;
    if (!el || !container) return;

    isScrollingRef.current = true;
    setActiveIndex(index);

    // Defer scroll to next frame so React's DOM commit from setActiveIndex
    // doesn't cancel the smooth scroll animation
    requestAnimationFrame(() => {
      const elRect = el.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const scrollTop = container.scrollTop + (elRect.top - containerRect.top) - 60;
      container.scrollTo({ top: Math.max(0, scrollTop), behavior: "smooth" });
    });

    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    scrollTimeoutRef.current = setTimeout(() => {
      isScrollingRef.current = false;
    }, 1200);
  }, [scrollContainerRef]);

  // Stable ref callbacks — avoids React nulling refs on every re-render
  const setSectionRef = useCallback((index) => {
    if (!refCallbacks.current.has(index)) {
      refCallbacks.current.set(index, (el) => {
        sectionRefs.current[index] = el;
      });
    }
    return refCallbacks.current.get(index);
  }, []);

  return { activeIndex, scrollToSection, setSectionRef };
}
