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
  { path: "/sources", component: () => import("../../pages/SourceRepositoriesList.vue") },
  {
    path: "/sources/:id",
    component: () => import("../../pages/SourceRepositoryDetail.vue"),
    props: true,
  },
  { path: "/compare", component: () => import("../../pages/ExperimentsComparison.vue") },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
