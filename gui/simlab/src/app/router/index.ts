import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  { path: "/", redirect: "/dashboard" },
  { path: "/dashboard", component: () => import("../../pages/Dashboard.vue") },
  { path: "/campaigns", component: () => import("../../pages/CampaignsList.vue") },
  {
    path: "/campaigns/:id",
    component: () => import("../../pages/CampaignDetail.vue"),
    props: true,
  },
  { path: "/experiments", component: () => import("../../pages/ExperimentsList.vue") },
  {
    path: "/experiments/:id",
    component: () => import("../../pages/ExperimentDetail.vue"),
    props: true,
  },
  { path: "/problems", component: () => import("../../pages/ProblemEditor.vue"), meta: { fullScreen: true } },
  { path: "/synthetic", component: () => import("../../pages/SyntheticEditor.vue"), meta: { fullScreen: true } },
  { path: "/sources", component: () => import("../../pages/SourceRepositoriesList.vue") },
  {
    path: "/sources/:id",
    component: () => import("../../pages/SourceRepositoryDetail.vue"),
    props: true,
  },
  { path: "/compare", component: () => import("../../pages/ExperimentsComparison.vue") },
  // Catch-all: unknown URLs render a 404 page instead of a blank shell
  { path: "/:pathMatch(.*)*", component: () => import("../../pages/NotFound.vue") },
];

export default createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(_to, _from, savedPosition) {
    // Restore the scroll position on back/forward, start at the top otherwise
    return savedPosition ?? { top: 0 };
  },
});
