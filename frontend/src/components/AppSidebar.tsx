import { useState } from "react";
import { LayoutDashboard, Camera, Play, Search, Info, Settings } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import { SettingsDialog } from "@/components/SettingsDialog";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";

const LOGO = "/branding/project-omnivisionx.png";

const navItems = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "Cameras", url: "/cameras", icon: Camera },
  { title: "Plate Search", url: "/plates", icon: Search },
  { title: "Simulator", url: "/simulator", icon: Play },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <>
      <Sidebar collapsible="icon">
        <SidebarHeader className="p-4">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <img
                src={LOGO}
                alt="OmniVisionX"
                className="h-9 w-9 rounded-lg object-contain border border-sidebar-border bg-background shrink-0"
              />
              <div className="min-w-0">
                <h1 className="text-sm font-semibold text-sidebar-foreground truncate">OmniVisionX</h1>
                <p className="text-[10px] text-muted-foreground">Traffic Analytics</p>
              </div>
            </div>
          )}
          {collapsed && (
            <div className="flex justify-center">
              <img
                src={LOGO}
                alt=""
                className="h-8 w-8 rounded-lg object-contain border border-sidebar-border bg-background"
              />
            </div>
          )}
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Navigation</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {navItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild isActive={location.pathname === item.url || (item.url !== "/" && location.pathname.startsWith(item.url))} tooltip={item.title}>
                      <NavLink
                        to={item.url}
                        end={item.url === "/"}
                        className="hover:bg-sidebar-accent/50"
                        activeClassName="bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                      >
                        <item.icon className="h-4 w-4" />
                        {!collapsed && <span>{item.title}</span>}
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarSeparator className="mx-2" />

        <SidebarFooter className="border-t border-sidebar-border pt-2 mt-auto">
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton
                asChild
                isActive={location.pathname === "/about"}
                tooltip="Giới thiệu"
              >
                <NavLink
                  to="/about"
                  className="hover:bg-sidebar-accent/50"
                  activeClassName="bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                >
                  <Info className="h-4 w-4" />
                  {!collapsed && <span>About</span>}
                </NavLink>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton
                type="button"
                onClick={() => setSettingsOpen(true)}
                tooltip="Cài đặt"
              >
                <Settings className="h-4 w-4" />
                {!collapsed && <span>Settings</span>}
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
      </Sidebar>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </>
  );
}
