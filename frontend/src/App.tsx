import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import { I18nProvider } from "@/i18n/I18nProvider";
import { AppLayout } from "@/components/AppLayout";
import Dashboard from "@/pages/Dashboard";
import CameraList from "@/pages/CameraList";
import CameraDetail from "@/pages/CameraDetail";
import Simulator from "@/pages/Simulator";
import PlateSearch from "@/pages/PlateSearch";
import About from "@/pages/About";
import LivePreview from "@/pages/LivePreview";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <ThemeProvider>
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route element={<AppLayout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/cameras" element={<CameraList />} />
                <Route path="/cameras/:id" element={<CameraDetail />} />
                <Route path="/live" element={<LivePreview />} />
                <Route path="/simulator" element={<Simulator />} />
                <Route path="/plates" element={<PlateSearch />} />
                <Route path="/about" element={<About />} />
              </Route>
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </I18nProvider>
  </ThemeProvider>
);

export default App;
