import { useState, useEffect, useRef, useCallback } from "react";

export function useActiveSection(sectionCount, scrollContainerRef) {
  const [activeIndex, setActiveIndex] = useState(0);
  const sectionRefs = useRef([]);
  const isScrollingRef = useRef(false);
  const scrollTimeoutRef = useRef(null);

  useEffect(() => {
    sectionRefs.current = sectionRefs.current.slice(0, sectionCount);
  }, [sectionCount]);

  // Track active section via IntersectionObserver
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (isScrollingRef.current) return;
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const idx = sectionRefs.current.indexOf(entry.target);
            if (idx !== -1) {
              setActiveIndex(idx);
            }
          }
        }
      },
      { threshold: 0.3, rootMargin: "-80px 0px -40% 0px" }
    );

    const currentRefs = sectionRefs.current;
    currentRefs.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => {
      currentRefs.forEach((ref) => {
        if (ref) observer.unobserve(ref);
      });
    };
  }, [sectionCount]);

  const scrollToSection = useCallback((index) => {
    const el = sectionRefs.current[index];
    if (!el) return;

    isScrollingRef.current = true;
    setActiveIndex(index);

    el.scrollIntoView({ behavior: "smooth", block: "start" });

    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    scrollTimeoutRef.current = setTimeout(() => {
      isScrollingRef.current = false;
    }, 800);
  }, []);

  const setSectionRef = useCallback((index) => (el) => {
    sectionRefs.current[index] = el;
  }, []);

  return { activeIndex, scrollToSection, setSectionRef };
}
